"""Esc 中断会话测试：TUI 按键发取消信号 + agent_worker 中断 turn。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcc_core import (
    Message,
    Notice,
    Role,
    TextBlock,
    ToolContext,
    ToolOutput,
    ToolResultBlock,
    ToolUseBlock,
    TurnEnd,
    Usage,
    UserEcho,
)
from mcc_harness import AgentLoopBuilder
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import Tool, ToolRegistry, default_registry
from mcc_tui import TuiHandles, _handle_key
from mcc_tui.app import App
from mcc_tui.keys import Key


def _handles():
    events: asyncio.Queue = asyncio.Queue()
    input_tx: asyncio.Queue = asyncio.Queue()
    cancel_tx: asyncio.Queue = asyncio.Queue()
    return (
        TuiHandles(
            events=events,
            input_tx=input_tx,
            session_short_id="t",
            cancel_tx=cancel_tx,
        ),
        cancel_tx,
    )


class TestEscCancelSignal:
    def test_escape_while_busy_sends_cancel(self):
        app = App(session_short_id="t")
        app.agent_busy = True
        handles, cancel_tx = _handles()
        _handle_key(app, handles, Key.escape)
        assert cancel_tx.get_nowait() is True
        notices = [it for it in app.items if it.kind == "notice"]
        assert any("取消" in it.text for it in notices)

    def test_escape_while_idle_no_cancel(self):
        app = App(session_short_id="t")
        app.agent_busy = False
        handles, cancel_tx = _handles()
        _handle_key(app, handles, Key.escape)
        assert cancel_tx.empty()

    def test_escape_busy_without_cancel_tx_still_notifies(self):
        app = App(session_short_id="t")
        app.agent_busy = True
        handles = TuiHandles(
            events=asyncio.Queue(),
            input_tx=asyncio.Queue(),
            session_short_id="t",
        )
        _handle_key(app, handles, Key.escape)
        notices = [it for it in app.items if it.kind == "notice"]
        assert any("取消" in it.text for it in notices)

    def test_other_keys_while_busy_no_cancel(self):
        app = App(session_short_id="t")
        app.agent_busy = True
        handles, cancel_tx = _handles()
        for key in ("a", Key.left, Key.up):
            _handle_key(app, handles, key)
        assert cancel_tx.empty()


class _SlowSource:
    """模拟长时间运行的 EventSource：yield UserEcho 后长时间阻塞。"""

    async def run(self, prompt: str):
        yield UserEcho(text=prompt)
        await asyncio.sleep(10)
        yield TurnEnd(cost_usd=0.0)


class _FastSource:
    """立即完成的 EventSource。"""

    async def run(self, prompt: str):
        yield UserEcho(text=prompt)
        yield TurnEnd(cost_usd=0.0)


def _patch_run_tui(captured: list, send_cancel: bool):
    """返回 fake run_tui：投递输入，可选发取消信号，收集事件到 captured。"""
    import mcc_tui

    original = mcc_tui.run_tui

    async def fake_run_tui(handles, console=None):
        handles.input_tx.put_nowait("task")
        if send_cancel:
            await asyncio.sleep(0.05)
            handles.cancel_tx.put_nowait(True)
        for _ in range(100):
            await asyncio.sleep(0.02)
            while not handles.events.empty():
                captured.append(handles.events.get_nowait())
            if any(isinstance(e, TurnEnd) for e in captured):
                break

    def install():
        mcc_tui.run_tui = fake_run_tui

    def restore():
        mcc_tui.run_tui = original

    return install, restore


@pytest.mark.asyncio
async def test_cancel_interrupts_running_turn(tmp_path: Path):
    import mcc_cli

    source = _SlowSource()
    captured: list = []
    install, restore = _patch_run_tui(captured, send_cancel=True)
    install()
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        restore()

    notices = [e for e in captured if isinstance(e, Notice)]
    turn_ends = [e for e in captured if isinstance(e, TurnEnd)]
    assert any("已中断" in n.text for n in notices)
    assert len(turn_ends) >= 1


@pytest.mark.asyncio
async def test_normal_turn_completes_without_cancel(tmp_path: Path):
    import mcc_cli

    source = _FastSource()
    captured: list = []
    install, restore = _patch_run_tui(captured, send_cancel=False)
    install()
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        restore()

    turn_ends = [e for e in captured if isinstance(e, TurnEnd)]
    assert len(turn_ends) == 1
    notices = [e for e in captured if isinstance(e, Notice)]
    assert not any("已中断" in n.text for n in notices)


@pytest.mark.asyncio
async def test_cancel_drains_pending_streaming_events(tmp_path: Path):
    """cancel 后已入队但未消费的流式事件应被清空，不残留闪现。"""
    import mcc_cli
    from mcc_core import TextDelta

    class BurstThenBlockSource:
        async def run(self, prompt: str):
            yield UserEcho(text=prompt)
            for i in range(20):
                yield TextDelta(text=f"chunk{i} ")
            await asyncio.sleep(10)
            yield TurnEnd(cost_usd=0.0)

    source = BurstThenBlockSource()
    captured: list = []
    install, restore = _patch_run_tui(captured, send_cancel=True)
    install()
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        restore()

    text_deltas = [e for e in captured if isinstance(e, TextDelta)]
    assert text_deltas == [], f"残留流式事件未清空: {text_deltas}"
    notices = [e for e in captured if isinstance(e, Notice)]
    assert any("已中断" in n.text for n in notices)


@pytest.mark.asyncio
async def test_stale_cancel_signal_does_not_break_next_turn(tmp_path: Path):
    """残留取消信号不应误中断下一个正常 turn。"""
    import mcc_cli

    class TwoTurnSource:
        def __init__(self):
            self._n = 0

        async def run(self, prompt: str):
            self._n += 1
            yield UserEcho(text=prompt)
            if self._n == 1:
                await asyncio.sleep(10)
            yield TurnEnd(cost_usd=0.0)

    source = TwoTurnSource()
    captured: list = []
    import mcc_tui

    original = mcc_tui.run_tui

    async def fake_run_tui(handles, console=None):
        handles.input_tx.put_nowait("first")
        await asyncio.sleep(0.05)
        handles.cancel_tx.put_nowait(True)
        handles.cancel_tx.put_nowait(True)
        for _ in range(100):
            await asyncio.sleep(0.02)
            while not handles.events.empty():
                captured.append(handles.events.get_nowait())
            if any(isinstance(e, TurnEnd) for e in captured):
                break
        handles.input_tx.put_nowait("second")
        for _ in range(100):
            await asyncio.sleep(0.02)
            while not handles.events.empty():
                captured.append(handles.events.get_nowait())
            turn_ends = [e for e in captured if isinstance(e, TurnEnd)]
            if len(turn_ends) >= 2:
                break

    mcc_tui.run_tui = fake_run_tui
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        mcc_tui.run_tui = original

    turn_ends = [e for e in captured if isinstance(e, TurnEnd)]
    assert len(turn_ends) == 2
    cancelled_notices = [e for e in captured if isinstance(e, Notice) and "已中断" in e.text]
    assert len(cancelled_notices) == 1


class _HangingTool:
    """执行时通知 started 事件后永久挂起，用于触发 cancel。"""

    def __init__(self, started: asyncio.Event) -> None:
        self._started = started

    @property
    def name(self) -> str:
        return "hang_forever"

    @property
    def description(self) -> str:
        return "test tool that hangs forever"

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        self._started.set()
        await asyncio.sleep(3600)
        return ToolOutput.ok("unreachable")


def _tool_use_turn(tool_id: str, tool_name: str, tool_input: dict) -> list:
    return [
        ToolUseStart(id=tool_id, name=tool_name),
        ToolUseInputDelta(partial_json=json.dumps(tool_input)),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


@pytest.fixture
def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="cancel-sess")


class TestCancelPreservesHistory:
    """cancel 后 agent.history 应保留被中断轮的上下文。"""

    async def test_cancel_during_tool_execution_finalizes_history(
        self, _ctx: ToolContext
    ):
        """工具执行中途 cancel：history 含 user + assistant(tool_use) + tool(interrupted)。"""
        started = asyncio.Event()
        registry = ToolRegistry()
        registry.register(_HangingTool(started))

        llm = MockLlm([_tool_use_turn("t1", "hang_forever", {})])
        agent = AgentLoopBuilder(llm, registry, _ctx).build()

        evq: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(agent.run_streaming("do something", evq))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        roles = [m.role for m in agent.history]
        assert roles == [Role.user, Role.assistant, Role.tool]

        assert agent.history[0].content[0].text == "do something"

        tool_uses = [
            b for b in agent.history[1].content if isinstance(b, ToolUseBlock)
        ]
        assert len(tool_uses) == 1
        assert tool_uses[0].id == "t1"

        tool_results = [
            b for b in agent.history[2].content if isinstance(b, ToolResultBlock)
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "t1"
        assert tool_results[0].is_error is True
        assert "interrupted" in tool_results[0].content

    async def test_cancel_during_tool_then_next_turn_works(
        self, _ctx: ToolContext
    ):
        """中断后下一轮能正常调用 LLM（消息序列合法）。"""
        started = asyncio.Event()
        registry = ToolRegistry()
        registry.register(_HangingTool(started))

        llm = MockLlm([
            _tool_use_turn("t1", "hang_forever", {}),
            single_text_turn("recovered after interrupt"),
        ])
        agent = AgentLoopBuilder(llm, registry, _ctx).build()

        evq: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(agent.run_streaming("first", evq))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        captured: list = []
        evq2: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("second", evq2)
        while not evq2.empty():
            captured.append(evq2.get_nowait())

        from mcc_core import TextDelta

        text_parts = [
            e.text for e in captured if isinstance(e, TextDelta)
        ]
        assert "".join(text_parts) != ""
        assert any(isinstance(e, TurnEnd) for e in captured)

    async def test_cancel_during_llm_stream_preserves_user_message(
        self, _ctx: ToolContext
    ):
        """LLM 流式输出中途 cancel：history 至少保留 user 消息。"""

        class _BlockingLlm:
            """stream 时通知 started 后挂起，模拟 LLM 响应中途被 cancel。"""

            def __init__(self, started: asyncio.Event) -> None:
                self._started = started

            async def complete(self, req):
                self._started.set()
                await asyncio.sleep(3600)

            def stream(self, req):
                self._started.set()

                async def _gen():
                    yield ToolUseStart(id="x", name="x")
                    await asyncio.sleep(3600)

                return _gen()

        started = asyncio.Event()
        llm = _BlockingLlm(started)
        agent = AgentLoopBuilder(llm, default_registry(), _ctx).build()

        evq: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(agent.run_streaming("hello", evq))
        await started.wait()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(agent.history) >= 1
        assert agent.history[0].role == Role.user
        assert agent.history[0].content[0].text == "hello"
