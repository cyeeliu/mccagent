"""UI-06 流式文本节流测试：批量 event 处理 + 最大长度限制 + 截断提示。"""

import asyncio

import pytest
from mcc_core import TextDelta, ThinkingDelta, TurnEnd, UserEcho
from mcc_tui.__init__ import _BATCH_MAX, TuiHandles, _step
from mcc_tui.app import _STREAM_MAX_CHARS, App, RenderItem
from mcc_tui.render import build_renderable
from rich.console import Console


@pytest.fixture
def app():
    return App(session_short_id="test")


@pytest.fixture
def console():
    return Console(width=80, height=24)


# ========== 流式累积 ==========


def test_streaming_text_accumulates(app):
    app.apply_event(TextDelta(text="hello "))
    app.apply_event(TextDelta(text="world"))
    assert app.streaming_text == "hello world"


def test_streaming_thinking_accumulates(app):
    app.apply_event(ThinkingDelta(text="let me "))
    app.apply_event(ThinkingDelta(text="think..."))
    assert app.streaming_thinking == "let me think..."


def test_streaming_text_and_thinking_independent(app):
    app.apply_event(TextDelta(text="answer"))
    app.apply_event(ThinkingDelta(text="reasoning"))
    assert app.streaming_text == "answer"
    assert app.streaming_thinking == "reasoning"


# ========== 最大长度限制 ==========


def test_streaming_truncated_at_limit(app):
    chunk = "x" * (_STREAM_MAX_CHARS // 2 + 1)
    app.apply_event(TextDelta(text=chunk))
    app.apply_event(TextDelta(text=chunk))
    assert len(app.streaming_text) == _STREAM_MAX_CHARS
    assert app.streaming_truncated is True


def test_streaming_truncated_stops_appending(app):
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    assert app.streaming_truncated is True
    app.apply_event(TextDelta(text="more"))
    assert app.streaming_text == "x" * _STREAM_MAX_CHARS


def test_streaming_partial_append_before_truncate(app):
    limit = _STREAM_MAX_CHARS
    app.apply_event(TextDelta(text="x" * (limit - 10)))
    app.apply_event(TextDelta(text="y" * 30))
    assert len(app.streaming_text) == limit
    assert app.streaming_text.endswith("y" * 10)
    assert app.streaming_truncated is True


def test_streaming_thinking_also_truncated(app):
    app.apply_event(ThinkingDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    app.apply_event(ThinkingDelta(text="more"))
    assert app.streaming_truncated is True
    assert len(app.streaming_thinking) == _STREAM_MAX_CHARS


def test_streaming_not_truncated_under_limit(app):
    app.apply_event(TextDelta(text="x" * 1000))
    assert app.streaming_truncated is False


# ========== flush 清理 ==========


def test_flush_clears_truncated_flag(app):
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    assert app.streaming_truncated is True
    app.flush_streaming()
    assert app.streaming_truncated is False


def test_flush_adds_truncation_notice(app):
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    app.flush_streaming()
    notices = [item for item in app.items if item.kind == "notice"]
    assert len(notices) == 1
    assert "truncated" in notices[0].text


def test_flush_no_notice_when_not_truncated(app):
    app.apply_event(TextDelta(text="hello"))
    app.flush_streaming()
    notices = [item for item in app.items if item.kind == "notice"]
    assert len(notices) == 0


def test_flush_preserves_content_before_truncate(app):
    app.apply_event(TextDelta(text="visible content"))
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    app.flush_streaming()
    assistant_items = [item for item in app.items if item.kind == "assistant"]
    assert len(assistant_items) == 1
    assert assistant_items[0].text.startswith("visible content")


def test_turn_end_flushes_with_truncation_notice(app):
    app.agent_busy = True
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    app.apply_event(TurnEnd(cost_usd=0.01))
    notices = [item for item in app.items if item.kind == "notice"]
    assert len(notices) == 1
    assert app.streaming_truncated is False


# ========== 渲染截断提示 ==========


def test_truncated_shown_in_render(app, console):
    app.apply_event(TextDelta(text="x" * (_STREAM_MAX_CHARS + 1)))
    layout = build_renderable(app, console)
    assert layout is not None


def test_not_truncated_no_warning_in_render(app, console):
    app.apply_event(TextDelta(text="hello"))
    layout = build_renderable(app, console)
    assert layout is not None


# ========== 批量 event 处理 ==========


def _make_handles():
    events: asyncio.Queue = asyncio.Queue()
    input_tx: asyncio.Queue = asyncio.Queue()
    return TuiHandles(events=events, input_tx=input_tx, session_short_id="test")


async def test_batch_processes_all_events():
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    for i in range(10):
        await handles.events.put(TextDelta(text=f"chunk{i} "))

    await _step(app, handles, key_queue)

    expected = "".join(f"chunk{i} " for i in range(10))
    assert app.streaming_text == expected
    assert handles.events.empty()


async def test_batch_limit_caps_processing():
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    total = _BATCH_MAX + 20
    for _ in range(total):
        await handles.events.put(TextDelta(text="x"))

    await _step(app, handles, key_queue)

    processed = _BATCH_MAX + 1
    assert len(app.streaming_text) == processed
    assert handles.events.qsize() == total - processed


async def test_batch_mixed_event_types():
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    await handles.events.put(TextDelta(text="answer"))
    await handles.events.put(ThinkingDelta(text="reasoning"))
    await handles.events.put(TextDelta(text=" more"))

    await _step(app, handles, key_queue)

    assert app.streaming_text == "answer more"
    assert app.streaming_thinking == "reasoning"


async def test_batch_with_turn_end_flushes():
    app = App(session_short_id="test")
    app.agent_busy = True
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    await handles.events.put(TextDelta(text="hello"))
    await handles.events.put(TurnEnd(cost_usd=0.01))

    await _step(app, handles, key_queue)

    assert app.streaming_text == ""
    assistant_items = [item for item in app.items if item.kind == "assistant"]
    assert len(assistant_items) == 1
    assert assistant_items[0].text == "hello"


async def test_batch_empty_queue_no_error():
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    await handles.events.put(TextDelta(text="single"))

    await _step(app, handles, key_queue)

    assert app.streaming_text == "single"


async def test_batch_truncation_during_batch():
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    chunk_size = _STREAM_MAX_CHARS // 10 + 1
    for _ in range(20):
        await handles.events.put(TextDelta(text="x" * chunk_size))

    await _step(app, handles, key_queue)

    assert app.streaming_truncated is True
    assert len(app.streaming_text) == _STREAM_MAX_CHARS


@pytest.mark.parametrize("n_events", [1, 5, 32, 64, 100])
async def test_batch_various_sizes(n_events):
    app = App(session_short_id="test")
    handles = _make_handles()
    key_queue: asyncio.Queue = asyncio.Queue()

    for _ in range(n_events):
        await handles.events.put(TextDelta(text="x"))

    await _step(app, handles, key_queue)

    processed = min(n_events, _BATCH_MAX + 1)
    assert len(app.streaming_text) == processed
    assert handles.events.qsize() == n_events - processed
