"""任务持久化领域模型：TaskRecord / TaskAttempt / TaskAssignment。

Phase 6 交付物：将 Task 领域模型从 mcc-tools 抽离到 mcc-core，
分离 task_id / attempt_id，引入 Assignment 持久化和 CAS 版本号。

每个正式任务有 TaskRecord；每次执行有独立 TaskAttempt；
每次分配有 TaskAssignment；重复派发不重复执行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.identifiers import (
    AgentId,
    AssignmentId,
    AttemptId,
    RunId,
    RuntimeId,
    TaskId,
)
from mcc_core.runtime.result import TokenUsage


class AttemptStatus(StrEnum):
    """单次执行尝试的状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class AssignmentStatus(StrEnum):
    """任务分配状态。"""

    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    COMPLETED = "completed"
    FAILED = "failed"


ATTEMPT_TERMINAL_STATUSES = frozenset({
    AttemptStatus.SUCCEEDED,
    AttemptStatus.FAILED,
    AttemptStatus.TIMED_OUT,
    AttemptStatus.CANCELLED,
})


def _now() -> datetime:
    return datetime.now(UTC)


def _gen_id(prefix: str) -> str:
    return f"{prefix}{uuid4().hex[:12]}"


@dataclass
class TaskRecord:
    """任务记录：持久化的任务元数据。

    每个正式任务有且仅有一个 TaskRecord，跨多次 Attempt 存在。
    version 用于 CAS（Compare-And-Swap）乐观锁。
    """

    task_id: TaskId
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    agent_id: AgentId | None = None
    run_id: RunId | None = None
    blocked_by: set[TaskId] = field(default_factory=set)
    priority: int = 0
    max_attempts: int = 3
    attempt_count: int = 0
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    deadline_at: datetime | None = None
    lease_expires_at: datetime | None = None
    error: str | None = None
    result_ref: str | None = None
    version: int = 0

    def touch(self) -> None:
        self.updated_at = _now()
        self.version += 1

    @property
    def can_retry(self) -> bool:
        return self.attempt_count < self.max_attempts

    @property
    def is_terminal(self) -> bool:
        return self.status in frozenset({
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        })


@dataclass
class TaskAttempt:
    """任务执行尝试：每次执行独立记录。

    同一 task_id 可有多个 TaskAttempt（重试），每个有独立 attempt_id。
    """

    attempt_id: AttemptId
    task_id: TaskId
    agent_id: AgentId
    runtime_id: RuntimeId | None = None
    status: AttemptStatus = AttemptStatus.PENDING
    idempotency_key: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    summary: str | None = None
    output: dict[str, object] | None = None
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    version: int = 0

    def __post_init__(self) -> None:
        if not self.idempotency_key:
            self.idempotency_key = f"{self.task_id}:{self.attempt_id}"

    def touch(self) -> None:
        self.version += 1

    @property
    def is_terminal(self) -> bool:
        return self.status in ATTEMPT_TERMINAL_STATUSES

    @property
    def is_retry(self) -> bool:
        return ":" in self.idempotency_key and not self.idempotency_key.endswith(
            f":{self.attempt_id}"
        )


@dataclass
class TaskAssignment:
    """任务分配记录：将任务分配给 agent 的持久化凭证。

    每次 dispatch 创建一个 TaskAssignment，支持 lease 过期回收。
    """

    assignment_id: AssignmentId
    task_id: TaskId
    attempt_id: AttemptId
    agent_id: AgentId
    runtime_id: RuntimeId | None = None
    status: AssignmentStatus = AssignmentStatus.ACTIVE
    assigned_at: datetime = field(default_factory=_now)
    lease_expires_at: datetime | None = None
    completed_at: datetime | None = None
    version: int = 0

    def touch(self) -> None:
        self.version += 1

    @property
    def is_expired(self) -> bool:
        if self.lease_expires_at is None:
            return False
        return _now() > self.lease_expires_at

    @property
    def is_active(self) -> bool:
        return self.status is AssignmentStatus.ACTIVE and not self.is_expired


__all__ = [
    "AttemptStatus",
    "AssignmentStatus",
    "ATTEMPT_TERMINAL_STATUSES",
    "TaskRecord",
    "TaskAttempt",
    "TaskAssignment",
]
