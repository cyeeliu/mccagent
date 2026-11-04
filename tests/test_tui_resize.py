"""UI-09 终端 resize 处理测试：ResizeWatcher + _step 集成。

验证 SIGWINCH/轮询检测、resize 触发缓存失效、resize 不丢失按键/事件。
"""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import patch

import pytest
from mcc_core import TextDelta
from mcc_tui.__init__ import TuiHandles, _step
from mcc_tui.app import App, RenderItem
from mcc_tui.resize import ResizeWatcher

# ========== ResizeWatcher 基础 ==========


def test_signal_callback_puts_to_queue():
    """_on_resize 向队列放入信号。"""
    rw = ResizeWatcher()
    rw._on_resize()
    assert not rw._queue.empty()
    assert rw._queue.get_nowait() is True


async def test_wait_receives_resize():
    """wait() 在 _on_resize 触发后返回。"""
    rw = ResizeWatcher()
    rw._on_resize()
    await rw.wait()


async def test_context_manager_starts_and_stops():
    """async with 自动 start/stop。"""
    rw = ResizeWatcher()
    async with rw:
        assert rw.using_signal or rw._poll_task is not None
    assert rw._poll_task is None or rw._poll_task.done()


async def test_using_signal_reflects_platform():
    """using_signal 在有 SIGWINCH 的 POSIX 平台为 True。"""
    rw = ResizeWatcher()
    rw.start()
    has_sigwinch = hasattr(signal, "SIGWINCH")
    assert rw.using_signal == has_sigwinch
    rw.stop()


# ========== 轮询路径 ==========


async def test_polling_detects_size_change():
    """轮询模式检测到终端尺寸变化。"""
    rw = ResizeWatcher(poll_interval=0.01)
    sizes = iter([
        (80, 24),
        (80, 24),
        (120, 40),
        (120, 40),
    ])

    def fake_get_terminal_size():
        cols, lines = next(sizes)
        return type("TermSize", (), {"columns": cols, "lines": lines})()

    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler", side_effect=NotImplementedError), \
         patch("mcc_tui.resize.os.get_terminal_size", side_effect=fake_get_terminal_size):
        rw.start()
        assert not rw.using_signal
        await rw.wait()
        await rw.wait()
        assert rw._last_size == (120, 40)
        rw.stop()


async def test_polling_ignores_oserror():
    """轮询遇到 OSError 时跳过，不崩溃。"""
    rw = ResizeWatcher(poll_interval=0.01)
    call_count = 0

    def fake_get_terminal_size():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise OSError("no tty")
        return type("TermSize", (), {"columns": 80, "lines": 24})()

    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler", side_effect=NotImplementedError), \
         patch("mcc_tui.resize.os.get_terminal_size", side_effect=fake_get_terminal_size):
        rw.start()
        await asyncio.wait_for(rw.wait(), timeout=1.0)
        rw.stop()


# ========== _step 集成：resize 清除缓存 ==========


def _make_handles():
    events: asyncio.Queue = asyncio.Queue()
    input_tx: asyncio.Queue = asyncio.Queue()
    return TuiHandles(events=events, input_tx=input_tx, session_short_id="test")


def _make_app_with_cache():
    """构造有渲染缓存数据的 App。"""
    app = App(session_short_id="test")
    item = RenderItem(kind="assistant", text="cached content")
    app.push(item)
    app.render_cache[item.seq] = "stale_entry"
    return app, item


async def test_step_clears_cache_on_resize():
    """resize 事件触发时清除 render_cache。"""
    app, item = _make_app_with_cache()
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    rw._on_resize()
    await _step(app, handles, key_queue, rw)

    assert app.render_cache == {}


async def test_resize_does_not_lose_key():
    """resize 和按键同时到达时，两者都被处理。"""
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    app.render_cache[999] = "stale"
    await key_queue.put("x")
    rw._on_resize()

    await _step(app, handles, key_queue, rw)

    assert app.render_cache == {}
    assert app.input == "x"
    assert app.cursor == 1


async def test_resize_does_not_lose_event():
    """resize 和 agent 事件同时到达时，两者都被处理。"""
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    app.render_cache[999] = "stale"
    await handles.events.put(TextDelta(text="hello"))
    rw._on_resize()

    await _step(app, handles, key_queue, rw)

    assert app.render_cache == {}
    assert app.streaming_text == "hello"


async def test_key_still_works_without_resize():
    """无 resize 时按键正常处理（回归测试）。"""
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    await key_queue.put("a")
    await _step(app, handles, key_queue, rw)

    assert app.input == "a"


async def test_event_still_works_without_resize():
    """无 resize 时事件正常处理（回归测试）。"""
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    await handles.events.put(TextDelta(text="data"))
    await _step(app, handles, key_queue, rw)

    assert app.streaming_text == "data"


# ========== 多次 resize ==========


async def test_multiple_resizes_clear_cache_each_time():
    """连续多次 resize 每次都清除缓存。"""
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()
    rw = ResizeWatcher()

    for _ in range(3):
        app.render_cache[999] = "stale"
        rw._on_resize()
        await _step(app, handles, key_queue, rw)
        assert app.render_cache == {}


# ========== stop 清理 ==========


async def test_stop_cancels_poll_task():
    """stop() 取消轮询任务。"""
    rw = ResizeWatcher(poll_interval=0.01)
    loop = asyncio.get_event_loop()
    with patch.object(loop, "add_signal_handler", side_effect=NotImplementedError), \
         patch("mcc_tui.resize.os.get_terminal_size", side_effect=OSError):
        rw.start()
        assert rw._poll_task is not None
        assert not rw._poll_task.done()
        rw.stop()
        assert rw._poll_task is None or rw._poll_task.done()


async def test_stop_is_idempotent():
    """多次调用 stop() 不报错。"""
    rw = ResizeWatcher()
    rw.start()
    rw.stop()
    rw.stop()
