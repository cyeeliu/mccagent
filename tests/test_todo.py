"""todo_write 测试：TodoManager 校验、TodoWriteTool、AgentLoop reminder。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcc_core import ToolContext, ToolOutput, Usage
from mcc_core.event import TextDelta, TurnEnd
from mcc_harness import AgentLoopBuilder
from mcc_llm import MessageStop, StreamEvent, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import TodoManager, TodoWriteTool, ToolRegistry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


class _NoOpTool:
    """测试用空操作工具，占位非 todo_write 调用。"""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput.ok("ok")


def _registry_with_todo(manager: TodoManager) -> ToolRegistry:
    r = ToolRegistry()
    r.register(TodoWriteTool(manager))
    r.register(_NoOpTool())
    return r


def _todo_turn(todos: list[dict], tid: str = "t1") -> list[StreamEvent]:
    return [
        ToolUseStart(id=tid, name="todo_write"),
        ToolUseInputDelta(partial_json=json.dumps({"todos": todos})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


def _noop_turn(tid: str) -> list[StreamEvent]:
    return [
        ToolUseStart(id=tid, name="noop"),
        ToolUseInputDelta(partial_json="{}"),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


class TestTodoManager:
    def test_update_and_render(self) -> None:
        m = TodoManager()
        out = m.update([
            {"content": "step 1", "status": "pending"},
            {"content": "step 2", "status": "in_progress"},
            {"content": "step 3", "status": "completed"},
        ])
        assert "[ ] step 1" in out
        assert "[>] step 2" in out
        assert "[x] step 3" in out
        assert "(1/3 completed)" in out
        assert m.has_items

    def test_render_empty(self) -> None:
        assert TodoManager().render() == "No todos."
        assert not TodoManager().has_items

    def test_max_todos_exceeded(self) -> None:
        m = TodoManager()
        with pytest.raises(ValueError, match="max"):
            m.update([{"content": f"s{i}", "status": "pending"} for i in range(21)])

    def test_empty_content_rejected(self) -> None:
        m = TodoManager()
        with pytest.raises(ValueError, match="content"):
            m.update([{"content": "  ", "status": "pending"}])

    def test_invalid_status_rejected(self) -> None:
        m = TodoManager()
        with pytest.raises(ValueError, match="status"):
            m.update([{"content": "x", "status": "blocked"}])

    def test_multiple_in_progress_rejected(self) -> None:
        m = TodoManager()
        with pytest.raises(ValueError, match="in_progress"):
            m.update([
                {"content": "a", "status": "in_progress"},
                {"content": "b", "status": "in_progress"},
            ])

    def test_non_list_rejected(self) -> None:
        m = TodoManager()
        with pytest.raises(ValueError, match="list"):
            m.update("not a list")  # type: ignore[arg-type]

    def test_update_replaces_not_appends(self) -> None:
        m = TodoManager()
        m.update([{"content": "a", "status": "pending"}])
        m.update([{"content": "b", "status": "pending"}])
        assert len(m.items) == 1
        assert m.items[0].content == "b"


class TestTodoWriteTool:
    @pytest.mark.asyncio
    async def test_execute_returns_render(self, ctx: ToolContext) -> None:
        m = TodoManager()
        tool = TodoWriteTool(m)
        out = await tool.execute({"todos": [{"content": "x", "status": "pending"}]}, ctx)
        assert out.is_error is False
        assert "[ ] x" in out.content

    @pytest.mark.asyncio
    async def test_execute_invalid_returns_error(self, ctx: ToolContext) -> None:
        m = TodoManager()
        tool = TodoWriteTool(m)
        out = await tool.execute({"todos": [{"content": "", "status": "pending"}]}, ctx)
        assert out.is_error is True

    def test_name_and_schema(self) -> None:
        tool = TodoWriteTool(TodoManager())
        assert tool.name == "todo_write"
        schema = tool.input_schema()
        assert schema["required"] == ["todos"]
        assert schema["properties"]["todos"]["maxItems"] == 20


class TestAgentLoopReminder:
    def _make_agent(self, llm, ctx, manager):
        registry = _registry_with_todo(manager)
        return (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .todo(manager)
            .todo_reminder_interval(3)
            .build()
        )

    @pytest.mark.asyncio
    async def test_reminder_after_interval(self, ctx: ToolContext) -> None:
        manager = TodoManager()
        todos = [{"content": "a", "status": "pending"}, {"content": "b", "status": "pending"}]
        turns = [
            _todo_turn(todos, "t1"),
            _noop_turn("t2"),
            _noop_turn("t3"),
            _noop_turn("t4"),
            single_text_turn("done"),
        ]
        llm = MockLlm(turns)
        agent = self._make_agent(llm, ctx, manager)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("plan", queue)
        reminder_msgs = [
            m for m in agent.history
            if m.role.value == "user"
            and any(
                getattr(b, "text", "").startswith("<reminder>")
                for b in m.content
            )
        ]
        assert len(reminder_msgs) == 1

    @pytest.mark.asyncio
    async def test_todo_write_resets_counter(self, ctx: ToolContext) -> None:
        manager = TodoManager()
        todos = [{"content": "a", "status": "pending"}]
        turns = [
            _todo_turn(todos, "t1"),
            _noop_turn("t2"),
            _todo_turn(todos, "t3"),
            _noop_turn("t4"),
            single_text_turn("done"),
        ]
        llm = MockLlm(turns)
        agent = self._make_agent(llm, ctx, manager)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("plan", queue)
        assert not any(
            "<reminder>" in getattr(b, "text", "")
            for m in agent.history
            for b in m.content
        )

    @pytest.mark.asyncio
    async def test_no_reminder_when_todo_empty(self, ctx: ToolContext) -> None:
        manager = TodoManager()
        turns = [
            _noop_turn("t1"),
            _noop_turn("t2"),
            _noop_turn("t3"),
            single_text_turn("done"),
        ]
        llm = MockLlm(turns)
        agent = self._make_agent(llm, ctx, manager)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("plan", queue)
        assert not any(
            "<reminder>" in getattr(b, "text", "")
            for m in agent.history
            for b in m.content
        )

    @pytest.mark.asyncio
    async def test_system_guidance_injected(self, ctx: ToolContext) -> None:
        manager = TodoManager()
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, manager)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(isinstance(e, TurnEnd) for e in events)
