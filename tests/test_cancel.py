"""Esc 中断会话测试：TUI 按键发取消信号 + agent_worker 中断 turn。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import Notice, TurnEnd, UserEcho
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
