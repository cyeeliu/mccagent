"""RuntimeEvent 统一事件体系（迁移方案 §11）。

Event 是通知，不是状态数据库。关键状态先写 TaskStore，再发事件。
sequence 用于检测丢失和乱序。TUI/CLI 订阅统一事件，不订阅 Runtime 私有事件。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mcc_core.runtime.enums import RuntimeEventType
from mcc_core.runtime.identifiers import (
    AgentId,
    AttemptId,
    CorrelationId,
    EventId,
    RuntimeId,
    TaskId,
)


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: EventId
    event_type: RuntimeEventType
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    runtime_id: RuntimeId = ""
    agent_id: AgentId = ""
    task_id: TaskId | None = None
    attempt_id: AttemptId | None = None
    sequence: int = 0
    correlation_id: CorrelationId | None = None
    causation_id: str | None = None
    result_ref: str | None = None
    schema_version: int = 1
    trace_id: str | None = None
    parent_event_id: str | None = None
    payload: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "runtime_id": self.runtime_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "sequence": self.sequence,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "result_ref": self.result_ref,
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "parent_event_id": self.parent_event_id,
            "payload": dict(self.payload),
        }


__all__ = ["RuntimeEvent"]
