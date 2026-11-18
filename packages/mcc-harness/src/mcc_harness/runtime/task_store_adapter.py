"""TaskStoreAdapter：将同步 mcc_tools.TaskStore 适配为异步 TaskStoreProtocol。

TaskDispatcher 需要异步 TaskStoreProtocol 接口，但 Orchestrator 历史上使用
同步 mcc_tools.TaskStore（文件持久化）。本适配器桥接两者：
- 任务级操作委托给 mcc_tools.TaskStore（同步调用包装为 async）
- Attempt 级操作使用内存存储（mcc_tools.TaskStore 无 Attempt 概念）
- 状态字符串与 TaskStatus 枚举互转
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.task_record import (
    AssignmentStatus,
    AttemptStatus,
    TaskAssignment,
    TaskAttempt,
    TaskRecord,
)

_STATUS_TO_ENUM: dict[str, TaskStatus] = {
    "pending": TaskStatus.PENDING,
    "claimed": TaskStatus.DISPATCHED,
    "dispatched": TaskStatus.DISPATCHED,
    "accepted": TaskStatus.ACCEPTED,
    "running": TaskStatus.RUNNING,
    "completed": TaskStatus.SUCCEEDED,
    "succeeded": TaskStatus.SUCCEEDED,
    "failed": TaskStatus.FAILED,
    "timeout": TaskStatus.TIMED_OUT,
    "timed_out": TaskStatus.TIMED_OUT,
    "cancelled": TaskStatus.CANCELLED,
}

_ENUM_TO_STATUS: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "pending",
    TaskStatus.QUEUED: "pending",
    TaskStatus.DISPATCHED: "claimed",
    TaskStatus.ACCEPTED: "claimed",
    TaskStatus.RUNNING: "claimed",
    TaskStatus.SUCCEEDED: "completed",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
    TaskStatus.TIMED_OUT: "timeout",
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


def _gen_attempt_id() -> str:
    import uuid

    return f"att_{uuid.uuid4().hex[:12]}"


class TaskStoreAdapter:
    """将同步 mcc_tools.TaskStore 适配为异步 TaskStoreProtocol。

    任务级操作委托给底层 TaskStore；Attempt 级操作使用内存存储。
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._attempts: dict[str, TaskAttempt] = {}
        self._task_attempts: dict[str, list[str]] = {}
        self._idempotency_index: dict[str, str] = {}
        self._assignments: dict[str, TaskAssignment] = {}
        self._lock = asyncio.Lock()

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
        task = self._store.create(subject, description)
        task.max_attempts = max_attempts
        if blocked_by:
            task.blockedBy = list(blocked_by)
        self._store.save(task)
        return _task_to_record(task)

    async def get_task(self, task_id: str) -> TaskRecord | None:
        try:
            task = self._store.load(task_id)
        except Exception:
            return None
        if task is None:
            return None
        return _task_to_record(task)

    async def list_tasks(
        self,
        *,
        run_id: str | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        tasks = self._store.list()
        records = [_task_to_record(t) for t in tasks]
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
        task = self._store.load(task_id)
        current = _STATUS_TO_ENUM.get(task.status, TaskStatus.PENDING)
        if current != from_status:
            return _task_to_record(task)
        new_status_str = _ENUM_TO_STATUS.get(to_status, to_status.value)
        if new_status_str != task.status:
            task.status = new_status_str
            if agent_id is not None:
                task.owner_agent_id = agent_id
                task.owner = agent_id
            task.version += 1
            self._store.save(task)
        return _task_to_record(task)

    async def claim_task(
        self,
        task_id: str,
        agent_id: str,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        task = self._store.claim(task_id, agent_id)
        return _task_to_record(task)

    async def renew_lease(
        self,
        task_id: str,
        agent_id: str,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        task = self._store.renew_lease(task_id, agent_id, lease_seconds=lease_seconds)
        return _task_to_record(task)

    async def expire_leases(self) -> list[str]:
        return []

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
            return attempt

    async def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        return self._attempts.get(attempt_id)

    async def list_attempts(self, task_id: str) -> list[TaskAttempt]:
        ids = self._task_attempts.get(task_id, [])
        return [self._attempts[i] for i in ids if i in self._attempts]

    async def complete_attempt(
        self,
        attempt_id: str,
        *,
        summary: str | None = None,
        output: dict[str, object] | None = None,
        usage: Any | None = None,
        cost_usd: float | None = None,
    ) -> TaskAttempt:
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(attempt_id)
            attempt.status = AttemptStatus.SUCCEEDED
            attempt.summary = summary
            attempt.finished_at = _now()
            return attempt

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
    ) -> TaskAttempt:
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                raise KeyError(attempt_id)
            attempt.status = AttemptStatus.FAILED
            attempt.error = error
            attempt.finished_at = _now()
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
        assignment = TaskAssignment(
            assignment_id=assignment_id,
            task_id=task_id,
            attempt_id=attempt_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
            status=AssignmentStatus.ACTIVE,
        )
        self._assignments[assignment_id] = assignment
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
        return assignment


__all__ = ["TaskStoreAdapter"]
