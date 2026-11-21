"""TaskStoreAdapter：将同步 mcc_tools.TaskStore 适配为异步 TaskStoreProtocol。

TaskDispatcher 需要异步 TaskStoreProtocol 接口，但 Orchestrator 历史上使用
同步 mcc_tools.TaskStore（文件持久化）。本适配器桥接两者：
- 任务级操作委托给 mcc_tools.TaskStore（同步调用包装为 async）
- Attempt/Assignment 级操作默认内存存储；提供 persist_dir 时持久化到 JSON 文件
- 状态字符串与 TaskStatus 枚举互转
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.errors import CASConflictError, InvalidTransitionError
from mcc_core.runtime.result import TokenUsage
from mcc_core.runtime.task_record import (
    AssignmentStatus,
    AttemptStatus,
    TaskAssignment,
    TaskAttempt,
    TaskRecord,
)

_STATUS_TO_ENUM: dict[str, TaskStatus] = {
    "pending": TaskStatus.PENDING,
    "queued": TaskStatus.QUEUED,
    "ready": TaskStatus.PENDING,
    "claimed": TaskStatus.DISPATCHED,
    "dispatched": TaskStatus.DISPATCHED,
    "accepted": TaskStatus.ACCEPTED,
    "running": TaskStatus.RUNNING,
    "in_progress": TaskStatus.RUNNING,
    "completed": TaskStatus.SUCCEEDED,
    "succeeded": TaskStatus.SUCCEEDED,
    "failed": TaskStatus.FAILED,
    "timeout": TaskStatus.TIMED_OUT,
    "timed_out": TaskStatus.TIMED_OUT,
    "cancelled": TaskStatus.CANCELLED,
}

_ENUM_TO_STATUS: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "pending",
    TaskStatus.QUEUED: "queued",
    TaskStatus.DISPATCHED: "dispatched",
    TaskStatus.ACCEPTED: "accepted",
    TaskStatus.RUNNING: "running",
    TaskStatus.SUCCEEDED: "succeeded",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
    TaskStatus.TIMED_OUT: "timed_out",
}


def _task_to_record(task: Any) -> TaskRecord:
    return TaskRecord(
        task_id=task.id,
        subject=task.subject,
        description=task.description,
        status=_STATUS_TO_ENUM.get(task.status, TaskStatus.PENDING),
        agent_id=task.owner_agent_id or task.owner,
        blocked_by=set(task.blockedBy) if task.blockedBy else set(),
        max_attempts=task.max_attempts or 3,
        attempt_count=task.attempt,
        version=task.version,
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _dt_to_iso(dt: datetime | None) -> str | None:
    """datetime 转 ISO 字符串，None 保持 None。"""
    return dt.isoformat() if dt else None


def _dt_from_iso(value: str | None) -> datetime | None:
    """ISO 字符串转 datetime，空值返回 None。"""
    return datetime.fromisoformat(value) if value else None


def _gen_attempt_id() -> str:
    import uuid

    return f"att_{uuid.uuid4().hex[:12]}"


class TaskStoreAdapter:
    """将同步 mcc_tools.TaskStore 适配为异步 TaskStoreProtocol。

    任务级操作委托给底层 TaskStore；Attempt/Assignment 级操作默认内存存储，
    提供 persist_dir 时持久化到 JSON 文件（原子写入），跨进程可恢复。
    """

    def __init__(
        self,
        store: Any,
        *,
        persist_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._persist_dir = persist_dir
        self._attempts: dict[str, TaskAttempt] = {}
        self._task_attempts: dict[str, list[str]] = {}
        self._idempotency_index: dict[str, str] = {}
        self._assignments: dict[str, TaskAssignment] = {}
        self._task_run_ids: dict[str, str] = {}
        self._lock = asyncio.Lock()
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_attempts()
            self._load_assignments()
            self._load_run_ids()

    # -- 持久化 --

    @staticmethod
    def _atomic_write(path: Path, data: str) -> None:
        """原子写入：写临时文件后 os.replace，避免崩溃留下半截 JSON。"""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _attempt_to_dict(attempt: TaskAttempt) -> dict[str, Any]:
        return {
            "attempt_id": attempt.attempt_id,
            "task_id": attempt.task_id,
            "agent_id": attempt.agent_id,
            "runtime_id": attempt.runtime_id,
            "status": attempt.status.value,
            "idempotency_key": attempt.idempotency_key,
            "started_at": _dt_to_iso(attempt.started_at),
            "finished_at": _dt_to_iso(attempt.finished_at),
            "error": attempt.error,
            "summary": attempt.summary,
            "output": attempt.output,
            "usage": attempt.usage.to_dict() if attempt.usage else None,
            "cost_usd": attempt.cost_usd,
            "artifacts": attempt.artifacts,
            "version": attempt.version,
        }

    @staticmethod
    def _attempt_from_dict(data: dict[str, Any]) -> TaskAttempt:
        usage = None
        if data.get("usage") is not None:
            usage = TokenUsage(**data["usage"])
        return TaskAttempt(
            attempt_id=data["attempt_id"],
            task_id=data["task_id"],
            agent_id=data["agent_id"],
            runtime_id=data.get("runtime_id"),
            status=AttemptStatus(data["status"]),
            idempotency_key=data.get("idempotency_key", ""),
            started_at=_dt_from_iso(data.get("started_at")),
            finished_at=_dt_from_iso(data.get("finished_at")),
            error=data.get("error"),
            summary=data.get("summary"),
            output=data.get("output"),
            usage=usage,
            cost_usd=data.get("cost_usd"),
            artifacts=data.get("artifacts"),
            version=data.get("version", 0),
        )

    @staticmethod
    def _assignment_to_dict(asgn: TaskAssignment) -> dict[str, Any]:
        return {
            "assignment_id": asgn.assignment_id,
            "task_id": asgn.task_id,
            "attempt_id": asgn.attempt_id,
            "agent_id": asgn.agent_id,
            "runtime_id": asgn.runtime_id,
            "status": asgn.status.value,
            "assigned_at": _dt_to_iso(asgn.assigned_at),
            "lease_expires_at": _dt_to_iso(asgn.lease_expires_at),
            "completed_at": _dt_to_iso(asgn.completed_at),
            "version": asgn.version,
        }

    @staticmethod
    def _assignment_from_dict(data: dict[str, Any]) -> TaskAssignment:
        return TaskAssignment(
            assignment_id=data["assignment_id"],
            task_id=data["task_id"],
            attempt_id=data["attempt_id"],
            agent_id=data["agent_id"],
            runtime_id=data.get("runtime_id"),
            status=AssignmentStatus(data["status"]),
            assigned_at=_dt_from_iso(data.get("assigned_at")) or _now(),
            lease_expires_at=_dt_from_iso(data.get("lease_expires_at")),
            completed_at=_dt_from_iso(data.get("completed_at")),
            version=data.get("version", 0),
        )

    def _load_attempts(self) -> None:
        """从 persist_dir 加载 attempts，并派生 task_attempts 与 idempotency 索引。"""
        if self._persist_dir is None:
            return
        path = self._persist_dir / "attempts.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in data:
            attempt = self._attempt_from_dict(item)
            self._attempts[attempt.attempt_id] = attempt
            self._task_attempts.setdefault(attempt.task_id, []).append(attempt.attempt_id)
            if attempt.idempotency_key:
                self._idempotency_index[attempt.idempotency_key] = attempt.attempt_id

    def _save_attempts(self) -> None:
        """将 attempts 持久化到 persist_dir（原子写入）。"""
        if self._persist_dir is None:
            return
        path = self._persist_dir / "attempts.json"
        payload = [self._attempt_to_dict(a) for a in self._attempts.values()]
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _load_assignments(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "assignments.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for item in data:
            asgn = self._assignment_from_dict(item)
            self._assignments[asgn.assignment_id] = asgn

    def _save_assignments(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "assignments.json"
        payload = [self._assignment_to_dict(a) for a in self._assignments.values()]
        self._atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _load_run_ids(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "run_ids.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._task_run_ids.update(data)

    def _save_run_ids(self) -> None:
        if self._persist_dir is None:
            return
        path = self._persist_dir / "run_ids.json"
        self._atomic_write(path, json.dumps(self._task_run_ids, ensure_ascii=False, indent=2))

    async def create_task(
        self,
        subject: str,
        description: str,
        *,
        run_id: str | None = None,
        blocked_by: set[str] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            task = self._store.create(subject, description)
            task.max_attempts = max_attempts
            if blocked_by:
                task.blockedBy = list(blocked_by)
            self._store.save(task)
            return _task_to_record(task)

        record = await asyncio.to_thread(_sync)
        if run_id is not None:
            self._task_run_ids[record.task_id] = run_id
            self._save_run_ids()
            record.run_id = run_id
        return record

    async def get_task(self, task_id: str) -> TaskRecord | None:
        def _sync() -> TaskRecord | None:
            try:
                task = self._store.load(task_id)
            except Exception:
                return None
            if task is None:
                return None
            return _task_to_record(task)

        record = await asyncio.to_thread(_sync)
        if record is not None and record.task_id in self._task_run_ids:
            record.run_id = self._task_run_ids[record.task_id]
        return record

    async def list_tasks(
        self,
        *,
        run_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        def _sync() -> list[TaskRecord]:
            tasks = self._store.list()
            return [_task_to_record(t) for t in tasks]

        records = await asyncio.to_thread(_sync)
        for r in records:
            if r.task_id in self._task_run_ids:
                r.run_id = self._task_run_ids[r.task_id]
        if run_id is not None:
            records = [r for r in records if r.run_id == run_id]
        if status is not None:
            records = [r for r in records if r.status == status]
        return records

    async def transition_task(
        self,
        task_id: str,
        from_status: TaskStatus,
        to_status: TaskStatus,
        *,
        agent_id: str | None = None,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            task = self._store.load(task_id)
            current = _STATUS_TO_ENUM.get(task.status, TaskStatus.PENDING)
            if current != from_status:
                raise CASConflictError(task_id, from_status.value, current.value)
            new_status_str = _ENUM_TO_STATUS.get(to_status, to_status.value)
            if new_status_str != task.status:
                task.status = new_status_str
                if agent_id is not None:
                    task.owner_agent_id = agent_id
                    task.owner = agent_id
                task.version += 1
                self._store.save(task)
            return _task_to_record(task)

        return await asyncio.to_thread(_sync)

    async def claim_task(
        self,
        task_id: str,
        agent_id: str,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            task = self._store.claim(task_id, agent_id, lease_seconds=lease_seconds)
            return _task_to_record(task)

        return await asyncio.to_thread(_sync)

    async def renew_lease(
        self,
        task_id: str,
        agent_id: str,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            task = self._store.renew_lease(task_id, agent_id, lease_seconds=lease_seconds)
            return _task_to_record(task)

        return await asyncio.to_thread(_sync)

    async def expire_leases(self) -> list[str]:
        """回收过期租约：将 lease_expires_at 已过期且非终态的任务重置为 PENDING。"""

        def _sync() -> list[str]:
            now = _now()
            expired: list[str] = []
            for task in self._store.list():
                lease = getattr(task, "lease_expires_at", None)
                if lease is None:
                    continue
                lease_dt: datetime | None
                if isinstance(lease, float | int):
                    lease_dt = datetime.fromtimestamp(lease, tz=UTC)
                elif isinstance(lease, datetime):
                    lease_dt = lease
                else:
                    continue
                if lease_dt < now and task.status in (
                    "claimed",
                    "dispatched",
                    "accepted",
                    "running",
                    "in_progress",
                ):
                    task.status = "pending"
                    task.owner_agent_id = None
                    task.owner = None
                    task.lease_expires_at = None
                    task.version += 1
                    self._store.save(task)
                    expired.append(task.id)
            return expired

        return await asyncio.to_thread(_sync)

    async def create_attempt(
        self,
        task_id: str,
        agent_id: str,
        *,
        runtime_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskAttempt:
        async with self._lock:
            if idempotency_key and idempotency_key in self._idempotency_index:
                existing_id = self._idempotency_index[idempotency_key]
                return self._attempts[existing_id]

            attempt_id = _gen_attempt_id()
            attempt = TaskAttempt(
                attempt_id=attempt_id,
                task_id=task_id,
                agent_id=agent_id,
                runtime_id=runtime_id,
                status=AttemptStatus.PENDING,
                idempotency_key=idempotency_key or "",
                started_at=_now(),
            )
            self._attempts[attempt_id] = attempt
            self._task_attempts.setdefault(task_id, []).append(attempt_id)
            if idempotency_key:
                self._idempotency_index[idempotency_key] = attempt_id
            # 递增底层 Task 的 attempt 计数器，与 InMemoryTaskStore 对齐
            try:
                task = self._store.load(task_id)
                if task is not None:
                    task.attempt += 1
                    task.version += 1
                    self._store.save(task)
            except Exception:
                pass
            self._save_attempts()
            return attempt

    async def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        return self._attempts.get(attempt_id)

    async def start_attempt(self, attempt_id: str) -> TaskAttempt:
        """将 PENDING attempt 转为 RUNNING 并持久化。"""
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt.status is not AttemptStatus.PENDING:
                raise InvalidTransitionError(
                    attempt.status.value,
                    AttemptStatus.RUNNING.value,
                )
            attempt.status = AttemptStatus.RUNNING
            attempt.touch()
            self._save_attempts()
            return attempt

    async def list_attempts(self, task_id: str) -> list[TaskAttempt]:
        ids = self._task_attempts.get(task_id, [])
        return [self._attempts[i] for i in ids if i in self._attempts]

    def _update_task_record_on_attempt_terminal(
        self,
        attempt: TaskAttempt,
    ) -> None:
        """Attempt 终态时同步更新底层 TaskRecord 状态。

        SUCCEEDED → TaskRecord.SUCCEEDED
        FAILED / TIMED_OUT / CANCELLED → TaskRecord.FAILED（当不可重试时）
        """
        task_status_str: str | None = None
        if attempt.status is AttemptStatus.SUCCEEDED:
            task_status_str = "completed"
        elif attempt.status in (
            AttemptStatus.FAILED,
            AttemptStatus.TIMED_OUT,
            AttemptStatus.CANCELLED,
        ):
            task_status_str = "failed"

        if task_status_str is None:
            return

        try:
            task = self._store.load(attempt.task_id)
        except Exception:
            return
        if task is None:
            return
        # 不可重试时才标记 TaskRecord 为终态
        if task_status_str == "failed":
            max_att = task.max_attempts or 3
            if task.attempt < max_att:
                return  # 仍可重试，不标记终态
        if task.status != task_status_str:
            task.status = task_status_str
            if attempt.error:
                task.error = attempt.error
            task.version += 1
            self._store.save(task)

    async def complete_attempt(
        self,
        attempt_id: str,
        *,
        summary: str | None = None,
        output: dict[str, object] | None = None,
        usage: Any | None = None,
        cost_usd: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> TaskAttempt:
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(attempt_id)
            # CAS 终态保护
            if attempt.is_terminal:
                if attempt.status is AttemptStatus.SUCCEEDED:
                    return attempt  # 幂等
                raise CASConflictError(
                    attempt_id, attempt.version, attempt.version,
                )
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.summary = summary
            attempt.output = output
            attempt.usage = usage
            attempt.cost_usd = cost_usd
            attempt.artifacts = artifacts
            attempt.finished_at = _now()
            attempt.touch()
            self._save_attempts()
            # 同步更新底层 TaskRecord
            self._update_task_record_on_attempt_terminal(attempt)
            return attempt

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> TaskAttempt:
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(attempt_id)
            if status not in (
                AttemptStatus.FAILED,
                AttemptStatus.TIMED_OUT,
                AttemptStatus.CANCELLED,
            ):
                raise InvalidTransitionError(AttemptStatus.RUNNING.value, status.value)
            # CAS 终态保护
            if attempt.is_terminal:
                if attempt.status is status:
                    return attempt  # 幂等
                raise CASConflictError(
                    attempt_id, attempt.version, attempt.version,
                )
            attempt.status = status
            attempt.error = error
            attempt.finished_at = _now()
            attempt.touch()
            self._save_attempts()
            # 同步更新底层 TaskRecord
            self._update_task_record_on_attempt_terminal(attempt)
            return attempt

    async def find_by_idempotency_key(self, key: str) -> TaskAttempt | None:
        attempt_id = self._idempotency_index.get(key)
        if attempt_id is None:
            return None
        return self._attempts.get(attempt_id)

    async def create_assignment(
        self,
        task_id: str,
        attempt_id: str,
        agent_id: str,
        *,
        runtime_id: str | None = None,
        lease_seconds: float = 300,
    ) -> TaskAssignment:
        import uuid

        assignment_id = f"asgn_{uuid.uuid4().hex[:12]}"
        now = _now()
        assignment = TaskAssignment(
            assignment_id=assignment_id,
            task_id=task_id,
            attempt_id=attempt_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
            status=AssignmentStatus.ACTIVE,
            assigned_at=now,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self._assignments[assignment_id] = assignment
        self._save_assignments()
        return assignment

    async def get_assignment(self, assignment_id: str) -> TaskAssignment | None:
        return self._assignments.get(assignment_id)

    async def list_assignments(
        self,
        *,
        task_id: str | None = None,
        agent_id: str | None = None,
        status: AssignmentStatus | None = None,
    ) -> list[TaskAssignment]:
        result = list(self._assignments.values())
        if task_id is not None:
            result = [a for a in result if a.task_id == task_id]
        if agent_id is not None:
            result = [a for a in result if a.agent_id == agent_id]
        if status is not None:
            result = [a for a in result if a.status == status]
        return result

    async def complete_assignment(self, assignment_id: str) -> TaskAssignment:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        assignment.status = AssignmentStatus.COMPLETED
        self._save_assignments()
        return assignment


__all__ = ["TaskStoreAdapter"]
