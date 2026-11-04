"""权限面板长文本分区测试 (UI-08)。

验证长消息文本不挤压操作提示，消息区域自动换行，操作提示固定保留。
"""

from __future__ import annotations

import pytest
from mcc_core import PermissionRequest
from mcc_tui.app import App
from mcc_tui.render import _permission_content_lines, _permission_panel


def _make_perm_app(message: str) -> App:
    """构造有 pending_permission 的 App。"""
    app = App(session_short_id="t")
    app.apply_event(
        PermissionRequest(id=1, message=message, tool_name="bash", tool_input={})
    )
    return app


def _panel_plain(panel) -> str:
    """从 Panel(Group(Text...)) 中提取纯文本。"""
    return "\n".join(t.plain for t in panel.renderable.renderables)


# -- 操作提示始终保留 --


def test_ops_present_short_message():
    """短消息时操作提示完整显示。"""
    app = _make_perm_app("Run npm install?")
    plain = _panel_plain(_permission_panel(app, 80))
    for token in ("[y]", "allow", "[a]", "always", "[n]", "deny", "[e]", "deny+comment"):
        assert token in plain


def test_ops_present_long_message():
    """长消息时操作提示仍完整显示，不被挤掉。"""
    app = _make_perm_app("A" * 200)
    plain = _panel_plain(_permission_panel(app, 60))
    for token in ("[y]", "allow", "[a]", "always", "[n]", "deny", "[e]", "deny+comment"):
        assert token in plain


def test_ops_on_last_line():
    """操作提示在消息之后的独立最后一行。"""
    app = _make_perm_app("Run command")
    lines = _panel_plain(_permission_panel(app, 80)).split("\n")
    assert "[y]" in lines[-1]
    assert "deny+comment" in lines[-1]


def test_empty_line_separates_message_and_ops():
    """消息和操作提示之间有空行分隔。"""
    app = _make_perm_app("Run command")
    lines = _panel_plain(_permission_panel(app, 80)).split("\n")
    assert len(lines) >= 3
    assert lines[-2] == ""


# -- 消息自动换行 --


def test_long_message_wraps():
    """长消息按面板宽度自动换行成多行。"""
    app = _make_perm_app("A" * 100)
    lines = _panel_plain(_permission_panel(app, 40)).split("\n")
    msg_lines = [line for line in lines[:-2] if line]
    assert len(msg_lines) > 1


def test_short_message_no_wrap():
    """短消息不换行。"""
    app = _make_perm_app("Run npm install?")
    lines = _panel_plain(_permission_panel(app, 80)).split("\n")
    msg_lines = [line for line in lines[:-2] if line]
    assert len(msg_lines) == 1


# -- 高度估算 --


@pytest.mark.parametrize("width", [80, 60, 40, 30])
def test_content_lines_matches_actual(width):
    """_permission_content_lines 与实际渲染行数一致。"""
    app = _make_perm_app("Run " + "B" * 80 + " now")
    plain = _panel_plain(_permission_panel(app, width))
    actual_lines = plain.count("\n") + 1
    estimated = _permission_content_lines(app, width)
    assert actual_lines == estimated, f"width={width}: actual={actual_lines}, estimated={estimated}"


def test_content_lines_short_message():
    """短消息时内容行数为 3（消息 + 空行 + 操作提示）。"""
    app = _make_perm_app("OK?")
    assert _permission_content_lines(app, 80) == 3


def test_content_lines_long_message():
    """长消息时内容行数大于 3。"""
    app = _make_perm_app("A" * 200)
    assert _permission_content_lines(app, 40) > 3


# -- 宽字符 --


def test_wide_char_message_wraps():
    """宽字符消息按显示宽度换行。"""
    app = _make_perm_app("中文" * 30)
    lines = _panel_plain(_permission_panel(app, 40)).split("\n")
    msg_lines = [line for line in lines[:-2] if line]
    assert len(msg_lines) > 1


# -- None 返回 --


def test_returns_none_without_permission():
    """无权限请求时返回 None。"""
    app = App(session_short_id="t")
    assert _permission_panel(app, 80) is None
    assert _permission_content_lines(app, 80) == 0
