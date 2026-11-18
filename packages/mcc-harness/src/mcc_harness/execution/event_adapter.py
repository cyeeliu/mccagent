"""AgentEvent → RuntimeEvent 适配器。

将 AgentLoop 产出的 mcc_core.AgentEvent 转换为统一 RuntimeEvent，
使 TUI/CLI 可以订阅统一事件流，不依赖 AgentLoop 私有事件类型。
"""

from __future__ import annotations

import uuid
from typing import Any

from mcc_core import AgentEvent
from mcc_core.runtime import RuntimeEvent, RuntimeEventType

_EVENT_TYPE_MAP: dict[type, RuntimeEventType] = {
    type(None): RuntimeEventType.TASK_PROGRESS,
}


def _map_event_type(event: AgentEvent) -> RuntimeEventType:
    from mcc_core.event import (
        Error,
        ToolCallEnd,
        ToolCallStart,
        TurnEnd,
    )

    if isinstance(event, ToolCallStart):
        return RuntimeEventType.TOOL_STARTED
    if isinstance(event, ToolCallEnd):
        return RuntimeEventType.TOOL_COMPLETED
    if isinstance(event, TurnEnd):
        return RuntimeEventType.TASK_PROGRESS
    if isinstance(event, Error):
        return RuntimeEventType.TASK_FAILED
    return RuntimeEventType.TASK_PROGRESS


def _extract_payload(event: AgentEvent) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(event):
        try:
            return asdict(event)
        except Exception:
            return {"type": type(event).__name__}
    return {"type": type(event).__name__}


class EventAdapter:
    def __init__(
        self,
        runtime_id: str,
        agent_id: str,
        task_id: str | None = None,
        attempt_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self._runtime_id = runtime_id
        self._agent_id = agent_id
        self._task_id = task_id
        self._attempt_id = attempt_id
        self._correlation_id = correlation_id
        self._sequence = 0

    def adapt(self, event: AgentEvent) -> RuntimeEvent:
        self._sequence += 1
        return RuntimeEvent(
            event_id=str(uuid.uuid4()),
            event_type=_map_event_type(event),
            runtime_id=self._runtime_id,
            agent_id=self._agent_id,
            task_id=self._task_id,
            attempt_id=self._attempt_id,
            sequence=self._sequence,
            correlation_id=self._correlation_id,
            payload=_extract_payload(event),
        )

    @property
    def sequence(self) -> int:
        return self._sequence


__all__ = ["EventAdapter"]
