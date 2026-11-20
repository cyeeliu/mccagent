"""TaskStore Protocol 与内存实现。

Phase 6 交付物：定义任务持久化的统一接口，从 mcc-tools 抽离。
InMemoryTaskStore 作为参考实现和测试桩。

设计要点：
- CAS 乐观锁：transition_task 通过 from_status 保证状态迁移原子性。
- 幂等 submit：find_by_idempotency_key 去重，同一 key 返回已有 Attempt。
- Lease 管理：claim_task / renew_lease / expire_leases。
- 结果先持久化再发事件：complete_attempt / fail_attempt 写入后才返回。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

from mcc_core.runtime.enums import TaskStatus, is_valid_task_transition
from mcc_core.runtime.errors import (
    CASConflictError,
    InvalidTransitionError,
    LeaseExpiredError,
    TaskNotFoundError,
)
from mcc_core.runtime.identifiers import AgentId, RunId, RuntimeId, TaskId
from mcc_core.runtime.result import TokenUsage
from mcc_core.runtime.task_record import (
    AssignmentStatus,
    AttemptStatus,
    TaskAssignment,
    TaskAttempt,
    TaskRecord,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _gen_task_id() -> str:
    from uuid import uuid4

    return f"task_{uuid4().hex[:12]}"


def _gen_attempt_id() -> str:
    from uuid import uuid4

    return f"att_{uuid4().hex[:12]}"


def _gen_assignment_id() -> str:
    from uuid import uuid4

    return f"asg_{uuid4().hex[:12]}"


@runtime_checkable
class TaskStoreProtocol(Protocol):
    """任务存储协议：定义任务持久化的统一接口。"""

    async def create_task(
        self,
        subject: str,
        description: str,
        *,
        run_id: RunId | None = None,
        blocked_by: set[TaskId] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> TaskRecord: ...

    async def get_task(self, task_id: TaskId) -> TaskRecord | None: ...

    async def list_tasks(
        self,
        *,
        run_id: RunId | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]: ...

    async def transition_task(
        self,
        task_id: TaskId,
        from_status: TaskStatus,
        to_status: TaskStatus,
        *,
        agent_id: AgentId | None = None,
    ) -> TaskRecord: ...

    async def claim_task(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord: ...

    async def renew_lease(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord: ...

    async def expire_leases(self) -> list[TaskId]: ...

    async def create_attempt(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        idempotency_key: str | None = None,
    ) -> TaskAttempt: ...

    async def get_attempt(self, attempt_id: str) -> TaskAttempt | None: ...

    async def start_attempt(self, attempt_id: str) -> TaskAttempt: ...

    async def list_attempts(self, task_id: TaskId) -> list[TaskAttempt]: ...

    async def complete_attempt(
        self,
        attempt_id: str,
        *,
        summary: str | None = None,
        output: dict[str, object] | None = None,
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> TaskAttempt: ...

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> TaskAttempt: ...

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> TaskAttempt | None: ...

    async def create_assignment(
        self,
        task_id: TaskId,
        attempt_id: str,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        lease_seconds: float = 300,
    ) -> TaskAssignment: ...

    async def get_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment | None: ...

    async def list_assignments(
        self,
        *,
        task_id: TaskId | None = None,
        agent_id: AgentId | None = None,
        status: AssignmentStatus | None = None,
    ) -> list[TaskAssignment]: ...

    async def complete_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment: ...


class InMemoryTaskStore:
    """内存任务存储：参考实现和测试桩。

    线程安全：asyncio.Lock 保护并发访问。
    CAS：transition_task 通过 from_status 校验保证原子性。
    幂等：create_attempt 检查 idempotency_key 去重。
    """

    def __init__(self) -> None:
        self._tasks: dict[TaskId, TaskRecord] = {}
        self._attempts: dict[str, TaskAttempt] = {}
        self._assignments: dict[str, TaskAssignment] = {}
        self._idempotency_index: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        subject: str,
        description: str,
        *,
        run_id: RunId | None = None,
        blocked_by: set[TaskId] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> TaskRecord:
        async with self._lock:
            task_id = _gen_task_id()
            task = TaskRecord(
                task_id=task_id,
                subject=subject,
                description=description,
                run_id=run_id,
                blocked_by=blocked_by or set(),
                priority=priority,
                max_attempts=max_attempts,
            )
            self._tasks[task_id] = task
            return task

    async def get_task(self, task_id: TaskId) -> TaskRecord | None:
        return self._tasks.get(task_id)

    async def list_tasks(
        self,
        *,
        run_id: RunId | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        result = list(self._tasks.values())
        if run_id is not None:
            result = [t for t in result if t.run_id == run_id]
        if status is not None:
            result = [t for t in result if t.status == status]
        return sorted(result, key=lambda t: t.priority)

    async def transition_task(
        self,
        task_id: TaskId,
        from_status: TaskStatus,
        to_status: TaskStatus,
        *,
        agent_id: AgentId | None = None,
    ) -> TaskRecord:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            if task.status != from_status:
                raise CASConflictError(task_id, from_status, task.version)
            if not is_valid_task_transition(task.status, to_status):
                raise InvalidTransitionError(str(task.status), str(to_status))
            task.status = to_status
            if agent_id is not None:
                task.agent_id = agent_id
            task.touch()
            return task

    async def claim_task(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            if task.status not in (TaskStatus.PENDING, TaskStatus.QUEUED):
                raise InvalidTransitionError(str(task.status), "claimed")
            task.status = TaskStatus.DISPATCHED
            task.agent_id = agent_id
            task.lease_expires_at = _now() + timedelta(seconds=lease_seconds)
            task.touch()
            return task

    async def renew_lease(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            if task.agent_id != agent_id:
                raise LeaseExpiredError(task_id)
            task.lease_expires_at = _now() + timedelta(seconds=lease_seconds)
            task.touch()
            return task

    async def expire_leases(self) -> list[TaskId]:
        async with self._lock:
            now = _now()
            expired: list[TaskId] = []
            for tid, task in self._tasks.items():
                if (
                    task.lease_expires_at is not None
                    and task.lease_expires_at < now
                    and task.status
                    in (TaskStatus.DISPATCHED, TaskStatus.ACCEPTED, TaskStatus.RUNNING)
                ):
                    task.status = TaskStatus.PENDING
                    task.agent_id = None
                    task.lease_expires_at = None
                    task.touch()
                    expired.append(tid)
            return expired

    async def create_attempt(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        idempotency_key: str | None = None,
    ) -> TaskAttempt:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)

            key = idempotency_key or ""
            if key:
                existing_id = self._idempotency_index.get(key)
                if existing_id is not None:
                    return self._attempts[existing_id]

            attempt_id = _gen_attempt_id()
            attempt = TaskAttempt(
                attempt_id=attempt_id,
                task_id=task_id,
                agent_id=agent_id,
                runtime_id=runtime_id,
                idempotency_key=key or f"{task_id}:{attempt_id}",
                status=AttemptStatus.PENDING,
                started_at=_now(),
            )
            self._attempts[attempt_id] = attempt
            if attempt.idempotency_key:
                self._idempotency_index[attempt.idempotency_key] = attempt_id
            task.attempt_count += 1
            task.touch()
            return attempt

    async def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        return self._attempts.get(attempt_id)

    async def start_attempt(self, attempt_id: str) -> TaskAttempt:
        """将 PENDING attempt 转为 RUNNING（PENDING → RUNNING 状态转换）。

        只有 PENDING attempt 可转换；非 PENDING 抛 InvalidTransitionError。
        """
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                from mcc_core.runtime.errors import AttemptNotFoundError

                raise AttemptNotFoundError(attempt_id)
            if attempt.status is not AttemptStatus.PENDING:
                raise InvalidTransitionError(
                    attempt.status.value,
                    AttemptStatus.RUNNING.value,
                )
            attempt.status = AttemptStatus.RUNNING
            attempt.touch()
            return attempt

    async def list_attempts(self, task_id: TaskId) -> list[TaskAttempt]:
        return [a for a in self._attempts.values() if a.task_id == task_id]

    async def complete_attempt(
        self,
        attempt_id: str,
        *,
        summary: str | None = None,
        output: dict[str, object] | None = None,
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> TaskAttempt:
        async with self._lock:
            attempt = self._attempts.get(attempt_id)
            if attempt is None:
                from mcc_core.runtime.errors import AttemptNotFoundError

                raise AttemptNotFoundError(attempt_id)
            # CAS 终态保护：终态不可被另一个终态覆盖
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

            task = self._tasks.get(attempt.task_id)
            if task is not None:
                task.status = TaskStatus.SUCCEEDED
                task.touch()
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
                from mcc_core.runtime.errors import AttemptNotFoundError

                raise AttemptNotFoundError(attempt_id)
            if status not in (
                AttemptStatus.FAILED,
                AttemptStatus.TIMED_OUT,
                AttemptStatus.CANCELLED,
            ):
                raise InvalidTransitionError(
                    AttemptStatus.RUNNING.value,
                    status.value,
                )
            # CAS 终态保护：终态不可被另一个终态覆盖
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

            task = self._tasks.get(attempt.task_id)
            if task is not None and not task.can_retry:
                task.status = TaskStatus.FAILED
                task.error = error
                task.touch()
            return attempt

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> TaskAttempt | None:
        attempt_id = self._idempotency_index.get(key)
        if attempt_id is None:
            return None
        return self._attempts.get(attempt_id)

    async def create_assignment(
        self,
        task_id: TaskId,
        attempt_id: str,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        lease_seconds: float = 300,
    ) -> TaskAssignment:
        async with self._lock:
            assignment = TaskAssignment(
                assignment_id=_gen_assignment_id(),
                task_id=task_id,
                attempt_id=attempt_id,
                agent_id=agent_id,
                runtime_id=runtime_id,
                lease_expires_at=_now() + timedelta(seconds=lease_seconds),
            )
            self._assignments[assignment.assignment_id] = assignment
            return assignment

    async def get_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment | None:
        return self._assignments.get(assignment_id)

    async def list_assignments(
        self,
        *,
        task_id: TaskId | None = None,
        agent_id: AgentId | None = None,
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

    async def complete_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment:
        async with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise TaskNotFoundError(assignment_id)
            assignment.status = AssignmentStatus.COMPLETED
            assignment.completed_at = _now()
            assignment.touch()
            return assignment


__all__ = [
    "TaskStoreProtocol",
    "InMemoryTaskStore",
]
