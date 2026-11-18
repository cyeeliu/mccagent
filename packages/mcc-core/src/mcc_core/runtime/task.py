"""RuntimeTask 与 TaskContext 领域模型。

RuntimeTask 是 Runtime.submit 的唯一输入类型（迁移方案 §3.3）。
task_id 在 Retry 期间保持不变，每次执行尝试生成新的 attempt_id。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mcc_core.runtime.errors import TaskValidationError
from mcc_core.runtime.identifiers import (
    AgentId,
    AssignmentId,
    AttemptId,
    RunId,
    TaskId,
)

_MAX_OBJECTIVE_LENGTH = 10_000


@dataclass(frozen=True)
class TaskContext:
    session_id: str | None = None
    parent_task_id: str | None = None
    working_directory: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    permissions: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RuntimeTask:
    task_id: TaskId
    attempt_id: AttemptId
    agent_id: AgentId
    objective: str
    input: Mapping[str, object] = field(default_factory=dict)
    run_id: RunId | None = None
    assignment_id: AssignmentId | None = None
    timeout_seconds: float | None = None
    idempotency_key: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    context: TaskContext = field(default_factory=TaskContext)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise TaskValidationError("task_id 不能为空")
        if not self.attempt_id:
            raise TaskValidationError("attempt_id 不能为空")
        if not self.agent_id:
            raise TaskValidationError("agent_id 不能为空")
        if not self.objective:
            raise TaskValidationError("objective 不能为空")
        if len(self.objective) > _MAX_OBJECTIVE_LENGTH:
            raise TaskValidationError(
                f"objective 长度超限 ({len(self.objective)} > {_MAX_OBJECTIVE_LENGTH})",
            )
        if not self.idempotency_key:
            object.__setattr__(self, "idempotency_key", f"{self.task_id}:{self.attempt_id}")

    @property
    def is_retry(self) -> bool:
        if ":" not in self.idempotency_key:
            return False
        return self.attempt_id != self.idempotency_key.split(":", 1)[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "agent_id": self.agent_id,
            "objective": self.objective,
            "input": dict(self.input),
            "run_id": self.run_id,
            "assignment_id": self.assignment_id,
            "timeout_seconds": self.timeout_seconds,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at.isoformat(),
            "context": {
                "session_id": self.context.session_id,
                "parent_task_id": self.context.parent_task_id,
                "working_directory": self.context.working_directory,
                "environment": dict(self.context.environment),
                "permissions": sorted(self.context.permissions),
            },
            "metadata": dict(self.metadata),
        }


__all__ = [
    "TaskContext",
    "RuntimeTask",
]
