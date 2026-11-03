"""TUI App 状态逻辑测试。"""

from __future__ import annotations

import json

import pytest
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
from mcc_tui.app import App, CallStatus, RenderItem
from mcc_tui.keys import Key


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


def test_selection_scroll_follows_cursor():
    """选项超过可视行数时，↓ 移动高亮行自动滚动窗口。"""
    import asyncio

    from mcc_tui import TuiHandles, _handle_key
    from mcc_tui.keys import Key
    from mcc_tui.render import _COMP_MAX_ROWS

    n = _COMP_MAX_ROWS + 5
    opts = [SelectionOption(label=f"opt{i}", value=f"v{i}") for i in range(n)]
    app = App(session_short_id="test")
    app.apply_event(SelectionRequest(id="sel-1", title="选择", options=opts))
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
    )

    for i in range(_COMP_MAX_ROWS):
        assert app.sel_index == i
        assert app.sel_scroll == 0
        _handle_key(app, handles, Key.down)

    assert app.sel_index == _COMP_MAX_ROWS
    assert app.sel_scroll == 1

    for _ in range(4):
        _handle_key(app, handles, Key.down)
    assert app.sel_index == n - 1
    assert app.sel_scroll == n - _COMP_MAX_ROWS

    _handle_key(app, handles, Key.down)
    assert app.sel_index == 0
    assert app.sel_scroll == 0


def test_selection_scroll_up_follows_cursor():
    """↑ 从顶部循环到末尾时，滚动窗口跳到底部。"""
    import asyncio

    from mcc_tui import TuiHandles, _handle_key
    from mcc_tui.keys import Key
    from mcc_tui.render import _COMP_MAX_ROWS

    n = _COMP_MAX_ROWS + 5
    opts = [SelectionOption(label=f"opt{i}", value=f"v{i}") for i in range(n)]
    app = App(session_short_id="test")
    app.apply_event(SelectionRequest(id="sel-1", title="选择", options=opts))
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
    )

    _handle_key(app, handles, Key.up)
    assert app.sel_index == n - 1
    assert app.sel_scroll == n - _COMP_MAX_ROWS


def _make_selection_app(n: int) -> tuple[App, object]:
    """构造 n 选项的选择浮层 App + handles。"""
    import asyncio

    from mcc_tui import TuiHandles

    opts = [SelectionOption(label=f"opt{i}", value=f"v{i}") for i in range(n)]
    app = App(session_short_id="test")
    app.apply_event(SelectionRequest(id="sel-1", title="选择", options=opts))
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="test",
    )
    return app, handles


@pytest.mark.parametrize("key,expected_index,expected_scroll", [
    (Key.home, 0, 0),
    (Key.end, 11, 5),
    (Key.page_down, 7, 1),
    (Key.page_up, 0, 0),
])
def test_selection_nav_keys(key, expected_index, expected_scroll):
    """Home/End/PageUp/PageDown 在选择浮层中正确定位并滚动。"""
    from mcc_tui import _handle_key
    from mcc_tui.keys import Key
    from mcc_tui.render import _COMP_MAX_ROWS

    n = _COMP_MAX_ROWS + 5
    app, handles = _make_selection_app(n)
    _handle_key(app, handles, key)
    assert app.sel_index == expected_index
    assert app.sel_scroll == expected_scroll


def test_selection_page_down_then_page_up():
    """PageDown 翻页后 PageUp 应回到起始位置。"""
    from mcc_tui import _handle_key
    from mcc_tui.keys import Key
    from mcc_tui.render import _COMP_MAX_ROWS

    n = _COMP_MAX_ROWS + 5
    app, handles = _make_selection_app(n)
    _handle_key(app, handles, Key.page_down)
    assert app.sel_index == _COMP_MAX_ROWS
    _handle_key(app, handles, Key.page_up)
    assert app.sel_index == 0
    assert app.sel_scroll == 0


def test_selection_page_down_clamps_to_last():
    """PageDown 不会超出最后一项。"""
    from mcc_tui import _handle_key
    from mcc_tui.keys import Key
    from mcc_tui.render import _COMP_MAX_ROWS

    n = _COMP_MAX_ROWS + 5
    app, handles = _make_selection_app(n)
    _handle_key(app, handles, Key.end)
    _handle_key(app, handles, Key.page_down)
    assert app.sel_index == n - 1


def test_selection_panel_height_stable():
    """选项超出可见区域时，面板内容行数恒定不随滚动变化。"""
    from mcc_tui.render import _COMP_MAX_ROWS, _selection_panel

    n = _COMP_MAX_ROWS + 5
    app, _ = _make_selection_app(n)

    line_counts: list[int] = []
    for scroll in range(n - _COMP_MAX_ROWS + 1):
        app.sel_scroll = scroll
        panel = _selection_panel(app)
        line_counts.append(panel.renderable.plain.count("\n"))

    assert len(set(line_counts)) == 1, (
        f"面板行数随滚动变化: {line_counts}"
    )
    assert line_counts[0] == _COMP_MAX_ROWS + 2


def test_selection_panel_no_indicators_when_fits():
    """选项不超出可见区域时，不显示指示行。"""
    from mcc_tui.render import _COMP_MAX_ROWS, _selection_panel

    n = _COMP_MAX_ROWS - 2
    app, _ = _make_selection_app(n)
    panel = _selection_panel(app)
    plain = panel.renderable.plain
    assert "更多" not in plain
    assert plain.count("\n") == n


@pytest.mark.parametrize("tool_name,arg_field,arg_value,expected", [
    ("spawn_subagent", "preset", "code_explorer", "code_explorer"),
    ("spawn_subagent", "preset", "test_runner", "test_runner"),
    ("spawn_teammate", "name", "worker1", "worker1"),
    ("send_teammate_message", "name", "worker1", "worker1"),
    ("read_file", "path", "foo.py", ""),
    ("run_bash", "cmd", "ls", ""),
])
def test_agent_name_extraction(tool_name, arg_field, arg_value, expected):
    """各种工具的 ToolCallStart 提取 agent_name。"""
    app = App(session_short_id="test")
    app.apply_event(ToolCallStart(
        id="call-1",
        name=tool_name,
        args_preview=json.dumps({arg_field: arg_value}),
    ))
    assert app.items[-1].agent_name == expected


def test_subagent_tool_result_inherits_agent_name():
    """ToolCallEnd 对应 spawn_subagent 时 tool_result 继承 agent_name。"""
    app = App(session_short_id="test")
    app.apply_event(ToolCallStart(
        id="call-1",
        name="spawn_subagent",
        args_preview='{"preset": "code_explorer", "task": "find files"}',
    ))
    app.apply_event(ToolCallEnd(
        id="call-1",
        output="[subagent=code_explorer tokens=100in/50out]\n找到 3 个文件",
        is_error=False,
    ))
    result_item = [it for it in app.items if it.kind == "tool_result"][0]
    assert result_item.agent_name == "code_explorer"


def test_subagent_render_shows_diamond():
    """subagent 的 tool_call 渲染包含 ◇ 标识而非工具名。"""
    from mcc_tui.render import _render_item
    from rich.console import Console

    item = RenderItem(kind="tool_call", name="spawn_subagent", agent_name="code_explorer")
    console = Console(record=True, width=80)
    console.print(_render_item(item))
    output = console.export_text()
    assert "◇" in output
    assert "code_explorer" in output
    assert "spawn_subagent" not in output


def test_subagent_result_skips_metadata():
    """subagent 的 tool_result 跳过 [subagent=...] 元数据行。"""
    from mcc_tui.render import _render_item
    from rich.console import Console

    item = RenderItem(
        kind="tool_result",
        agent_name="code_explorer",
        output="[subagent=code_explorer tokens=100in/50out]\n找到 3 个文件",
    )
    console = Console(record=True, width=80)
    console.print(_render_item(item))
    output = console.export_text()
    assert "找到 3 个文件" in output
    assert "[subagent=" not in output


def test_normal_tool_render_unchanged():
    """普通工具渲染不受 agent_name 影响：显示 ▶ 和工具名。"""
    from mcc_tui.render import _render_item
    from rich.console import Console

    item = RenderItem(kind="tool_call", name="read_file", args_preview='{"path": "foo.py"}')
    console = Console(record=True, width=80)
    console.print(_render_item(item))
    output = console.export_text()
    assert "▶" in output
    assert "read_file" in output
    assert "◇" not in output
