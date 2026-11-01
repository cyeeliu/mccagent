"""TUI App 状态逻辑测试。"""

from __future__ import annotations

from mcc_core import (
    Error as ErrorEvent,
)
from mcc_core import (
    Notice,
    PermissionRequest,
    SelectionOption,
    SelectionRequest,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_tui.app import App, CallStatus


def test_accumulates_text_deltas():
    app = App(session_short_id="test")
    app.apply_event(UserEcho(text="hi"))
    app.apply_event(TextDelta(text="he"))
    app.apply_event(TextDelta(text="llo"))
    assert app.streaming_text == "hello"


def test_flushes_on_turn_end():
    app = App(session_short_id="test")
    app.agent_busy = True
    app.apply_event(TextDelta(text="response"))
    app.apply_event(TurnEnd(cost_usd=0.01))
    assert app.streaming_text == ""
    assert not app.agent_busy
    assert app.cost_usd == 0.01
    last = app.items[-1]
    assert last.kind == "assistant"
    assert last.text == "response"


def test_tool_call_updates_status_done():
    app = App(session_short_id="test")
    app.apply_event(ToolCallStart(id="c1", name="read", args_preview="{}"))
    app.apply_event(ToolCallEnd(id="c1", output="42", is_error=False))
    call_items = [it for it in app.items if it.kind == "tool_call"]
    assert any(it.status == CallStatus.done for it in call_items)


def test_tool_call_updates_status_failed():
    app = App(session_short_id="test")
    app.apply_event(ToolCallStart(id="c1", name="read", args_preview="{}"))
    app.apply_event(ToolCallEnd(id="c1", output="boom", is_error=True))
    call_items = [it for it in app.items if it.kind == "tool_call"]
    assert any(it.status == CallStatus.failed for it in call_items)
    result_items = [it for it in app.items if it.kind == "tool_result"]
    assert result_items[0].is_error is True


def test_error_sets_not_busy():
    app = App(session_short_id="test")
    app.agent_busy = True
    app.apply_event(ErrorEvent(text="something broke"))
    assert not app.agent_busy
    assert app.items[-1].kind == "error"


def test_permission_request_sets_pending():
    app = App(session_short_id="test")
    app.apply_event(PermissionRequest(id=1, message="allow bash?"))
    assert app.pending_permission is not None
    assert app.pending_permission.id == 1


def test_permission_requests_queued():
    app = App(session_short_id="test")
    app.apply_event(PermissionRequest(id=1, message="first?"))
    app.apply_event(PermissionRequest(id=2, message="second?"))
    assert app.pending_permission.id == 1
    assert [p.id for p in app.permission_queue] == [2]


def test_resolve_permission_pops_next():
    app = App(session_short_id="test")
    app.apply_event(PermissionRequest(id=1, message="first?"))
    app.apply_event(PermissionRequest(id=2, message="second?"))
    resolved = app.resolve_permission()
    assert resolved is not None and resolved.id == 1
    assert app.pending_permission is not None
    assert app.pending_permission.id == 2
    assert app.permission_queue == []


def test_notice_added():
    app = App(session_short_id="test")
    app.apply_event(Notice(text="info msg"))
    assert app.items[-1].kind == "notice"
    assert app.items[-1].text == "info msg"


def test_max_items_cap():
    app = App(session_short_id="test")
    for i in range(600):
        app.apply_event(Notice(text=str(i)))
    assert len(app.items) <= 500


def test_flush_streaming_before_tool_call():
    app = App(session_short_id="test")
    app.apply_event(TextDelta(text="partial"))
    app.apply_event(ToolCallStart(id="c1", name="bash", args_preview="ls"))
    assert app.streaming_text == ""
    assert app.items[0].kind == "assistant"
    assert app.items[1].kind == "tool_call"


# -- SelectionRequest（/resume 选择浮层） --


def test_selection_request_sets_pending():
    app = App(session_short_id="test")
    req = SelectionRequest(
        id="sel-1",
        title="选择会话",
        options=[SelectionOption(label="a", value="v1"), SelectionOption(label="b", value="v2")],
    )
    app.apply_event(req)
    assert app.pending_selection is not None
    assert app.pending_selection.id == "sel-1"
    assert app.sel_index == 0


def test_selection_key_navigation():
    """↑↓ 在候选列表内循环移动高亮行。"""
    import asyncio

    from mcc_tui import TuiHandles, _handle_key
    from mcc_tui.keys import Key

    app = App(session_short_id="test")
    req = SelectionRequest(
        id="sel-1",
        title="选择",
        options=[SelectionOption(label="a", value="v1"), SelectionOption(label="b", value="v2")],
    )
    app.apply_event(req)
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
    )
    _handle_key(app, handles, Key.down)
    assert app.sel_index == 1
    _handle_key(app, handles, Key.down)
    assert app.sel_index == 0
    _handle_key(app, handles, Key.up)
    assert app.sel_index == 1


def test_selection_enter_sends_decision():
    """Enter 回传选中项的 value 并关闭浮层。"""
    import asyncio

    from mcc_core import SelectionDecision
    from mcc_tui import TuiHandles, _handle_key
    from mcc_tui.keys import Key

    app = App(session_short_id="test")
    req = SelectionRequest(
        id="sel-1",
        title="选择",
        options=[SelectionOption(label="a", value="v1"), SelectionOption(label="b", value="v2")],
    )
    app.apply_event(req)
    app.sel_index = 1
    sel_tx: asyncio.Queue = asyncio.Queue()
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
        selection_tx=sel_tx,
    )
    _handle_key(app, handles, Key.enter)
    decision = sel_tx.get_nowait()
    assert isinstance(decision, SelectionDecision)
    assert decision.request_id == "sel-1"
    assert decision.value == "v2"
    assert app.pending_selection is None
    assert app.sel_index == 0


def test_selection_escape_cancels():
    """Esc 回传 value=None 并关闭浮层。"""
    import asyncio

    from mcc_core import SelectionDecision
    from mcc_tui import TuiHandles, _handle_key
    from mcc_tui.keys import Key

    app = App(session_short_id="test")
    req = SelectionRequest(
        id="sel-1",
        title="选择",
        options=[SelectionOption(label="a", value="v1")],
    )
    app.apply_event(req)
    sel_tx: asyncio.Queue = asyncio.Queue()
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
        selection_tx=sel_tx,
    )
    _handle_key(app, handles, Key.escape)
    decision = sel_tx.get_nowait()
    assert decision.value is None
    assert app.pending_selection is None
