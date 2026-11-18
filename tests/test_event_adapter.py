"""Phase 2: EventAdapter 测试。

覆盖：AgentEvent → RuntimeEvent 映射、sequence 递增、payload 提取、
correlation_id 传递、不同事件类型映射。
"""

from __future__ import annotations

import pytest
from mcc_core.event import (
    Error,
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_core.runtime import RuntimeEvent, RuntimeEventType
from mcc_harness.execution.event_adapter import EventAdapter


class TestEventAdapterBasic:
    def test_adapt_returns_runtime_event(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(UserEcho(text="hello"))
        assert isinstance(event, RuntimeEvent)

    def test_adapt_preserves_ids(self):
        adapter = EventAdapter(
            runtime_id="rt-1",
            agent_id="agent-1",
            task_id="task-1",
            attempt_id="att-1",
            correlation_id="corr-1",
        )
        event = adapter.adapt(TextDelta(text="x"))
        assert event.runtime_id == "rt-1"
        assert event.agent_id == "agent-1"
        assert event.task_id == "task-1"
        assert event.attempt_id == "att-1"
        assert event.correlation_id == "corr-1"

    def test_adapt_generates_unique_event_id(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        e1 = adapter.adapt(TextDelta(text="a"))
        e2 = adapter.adapt(TextDelta(text="b"))
        assert e1.event_id != e2.event_id

    def test_optional_ids_default_none(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(TextDelta(text="x"))
        assert event.task_id is None
        assert event.attempt_id is None
        assert event.correlation_id is None


class TestEventAdapterSequence:
    def test_sequence_starts_at_zero(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        assert adapter.sequence == 0

    def test_sequence_increments(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        adapter.adapt(TextDelta(text="a"))
        assert adapter.sequence == 1
        adapter.adapt(TextDelta(text="b"))
        assert adapter.sequence == 2
        adapter.adapt(TextDelta(text="c"))
        assert adapter.sequence == 3

    def test_event_sequence_matches_property(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        e1 = adapter.adapt(TextDelta(text="a"))
        e2 = adapter.adapt(TextDelta(text="b"))
        e3 = adapter.adapt(TextDelta(text="c"))
        assert e1.sequence == 1
        assert e2.sequence == 2
        assert e3.sequence == 3


class TestEventAdapterTypeMapping:
    @pytest.mark.parametrize("event,expected_type", [
        (ToolCallStart(id="t1", name="bash", args_preview="ls"), RuntimeEventType.TOOL_STARTED),
        (ToolCallEnd(id="t1", output="done", is_error=False), RuntimeEventType.TOOL_COMPLETED),
        (ToolCallEnd(id="t1", output="err", is_error=True), RuntimeEventType.TOOL_COMPLETED),
        (TurnEnd(cost_usd=0.01), RuntimeEventType.TASK_PROGRESS),
        (Error(text="boom"), RuntimeEventType.TASK_FAILED),
        (UserEcho(text="hi"), RuntimeEventType.TASK_PROGRESS),
        (TextDelta(text="chunk"), RuntimeEventType.TASK_PROGRESS),
        (Notice(text="note"), RuntimeEventType.TASK_PROGRESS),
    ])
    def test_event_type_mapping(self, event, expected_type):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        result = adapter.adapt(event)
        assert result.event_type == expected_type


class TestEventAdapterPayload:
    def test_payload_from_dataclass(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(ToolCallStart(id="t1", name="bash", args_preview="ls"))
        assert event.payload["id"] == "t1"
        assert event.payload["name"] == "bash"
        assert event.payload["args_preview"] == "ls"

    def test_payload_from_text_delta(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(TextDelta(text="hello world"))
        assert event.payload["text"] == "hello world"

    def test_payload_from_error(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(Error(text="something broke"))
        assert event.payload["text"] == "something broke"

    def test_payload_from_turn_end(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(TurnEnd(cost_usd=0.05))
        assert event.payload["cost_usd"] == 0.05


class TestEventAdapterOccurredAt:
    def test_event_has_timestamp(self):
        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        event = adapter.adapt(TextDelta(text="x"))
        assert event.occurred_at is not None

    def test_events_have_increasing_or_equal_timestamps(self):
        import time

        adapter = EventAdapter(runtime_id="rt-1", agent_id="agent-1")
        e1 = adapter.adapt(TextDelta(text="a"))
        time.sleep(0.001)
        e2 = adapter.adapt(TextDelta(text="b"))
        assert e2.occurred_at >= e1.occurred_at
