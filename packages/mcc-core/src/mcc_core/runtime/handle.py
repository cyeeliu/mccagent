"""TaskHandle：Runtime 已接受任务的凭证。

TaskHandle 表示 Runtime 已接受任务，不表示任务已完成。
等待结果应通过 TaskStore / ResultStore 查询（迁移方案 §3.4）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from mcc_core.runtime.identifiers import AttemptId, RuntimeId, TaskId


@dataclass(frozen=True)
class TaskHandle:
    task_id: TaskId
    attempt_id: AttemptId
    runtime_id: RuntimeId
    accepted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "runtime_id": self.runtime_id,
            "accepted_at": self.accepted_at.isoformat(),
        }


__all__ = ["TaskHandle"]
