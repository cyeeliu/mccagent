"""补全面板滚动与高度估算测试 (UI-07)。

验证候选项超出可见区域时的滑动窗口、指示行、滚动同步和高度估算。
"""

from __future__ import annotations

import asyncio

import pytest
from mcc_tui import TuiHandles, _handle_key
from mcc_tui.app import App
from mcc_tui.completions import CompletionItem, make_provider
from mcc_tui.keys import Key
from mcc_tui.render import _COMP_MAX_ROWS, _PANEL_BORDERS, _completion_panel


def _make_comp_app(n: int) -> tuple[App, TuiHandles]:
    """构造 n 候选的补全浮层 App + handles。"""
    commands = [(f"cmd{i}", f"hint {i}") for i in range(n)]
    provider = make_provider(commands)
    app = App(session_short_id="t")
    app.input = "/"
    app.refresh_completions(provider)
    handles = TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="t",
        completions=provider,
    )
    return app, handles


# -- 面板高度恒定 --


def test_panel_height_stable_when_overflow():
    """候选超出可见区域时，面板内容行数恒定不随滚动变化。"""
    n = _COMP_MAX_ROWS + 5
    app, _ = _make_comp_app(n)

    line_counts: list[int] = []
    for scroll in range(n - _COMP_MAX_ROWS + 1):
        app.comp_scroll = scroll
        panel = _completion_panel(app, 80)
        line_counts.append(panel.renderable.plain.count("\n"))

    assert len(set(line_counts)) == 1, f"面板行数随滚动变化: {line_counts}"
    assert line_counts[0] == _COMP_MAX_ROWS + 2


def test_no_indicators_when_fits():
    """候选不超出可见区域时，不显示指示行。"""
    n = _COMP_MAX_ROWS - 2
    app, _ = _make_comp_app(n)
    panel = _completion_panel(app, 80)
    plain = panel.renderable.plain
    assert "更多" not in plain
    assert plain.count("\n") == n


def test_indicators_shown_when_overflow():
    """候选超出可见区域时，在中间滚动位置显示上下指示行。"""
    n = _COMP_MAX_ROWS + 3
    app, _ = _make_comp_app(n)
    app.comp_scroll = 1
    panel = _completion_panel(app, 80)
    plain = panel.renderable.plain
    assert "↑ 更多" in plain
    assert "↓ 更多" in plain


def test_indicator_up_hidden_at_top():
    """滚动位置在顶部时，上指示行用空行占位。"""
    n = _COMP_MAX_ROWS + 3
    app, _ = _make_comp_app(n)
    app.comp_scroll = 0
    panel = _completion_panel(app, 80)
    plain = panel.renderable.plain
    assert "↑ 更多" not in plain
    assert "↓ 更多" in plain


def test_indicator_down_hidden_at_bottom():
    """滚动位置在底部时，下指示行用空行占位。"""
    n = _COMP_MAX_ROWS + 3
    app, _ = _make_comp_app(n)
    app.comp_scroll = n - _COMP_MAX_ROWS
    panel = _completion_panel(app, 80)
    plain = panel.renderable.plain
    assert "↑ 更多" in plain
    assert "↓ 更多" not in plain


# -- 高度估算与实际一致 --


@pytest.mark.parametrize("n", [
    1,
    _COMP_MAX_ROWS - 1,
    _COMP_MAX_ROWS,
    _COMP_MAX_ROWS + 1,
    _COMP_MAX_ROWS + 5,
])
def test_comp_rows_matches_actual(n):
    """comp_rows 估算与面板实际渲染总高度一致。"""
    app, _ = _make_comp_app(n)
    panel = _completion_panel(app, 80)
    actual_content = panel.renderable.plain.count("\n")
    actual_total = actual_content + _PANEL_BORDERS
    visible = min(_COMP_MAX_ROWS, n)
    indicators = 2 if n > visible else 0
    estimated = visible + indicators + _PANEL_BORDERS
    assert actual_total == estimated, f"n={n}: actual={actual_total}, estimated={estimated}"


# -- 滚动同步 --


@pytest.mark.parametrize("key,expected_index,expected_scroll", [
    (Key.home, 0, 0),
    (Key.end, 11, 5),
    (Key.page_down, 7, 1),
    (Key.page_up, 0, 0),
])
def test_nav_keys(key, expected_index, expected_scroll):
    """Home/End/PageUp/PageDown 在补全浮层中正确定位并滚动。"""
    n = _COMP_MAX_ROWS + 5
    app, handles = _make_comp_app(n)
    _handle_key(app, handles, key)
    assert app.comp_index == expected_index
    assert app.comp_scroll == expected_scroll


def test_down_scrolls_when_exceeding_visible():
    """连续按 ↓ 超出可见区域时，comp_scroll 同步下移。"""
    n = _COMP_MAX_ROWS + 3
    app, handles = _make_comp_app(n)
    for _ in range(_COMP_MAX_ROWS):
        _handle_key(app, handles, Key.down)
    assert app.comp_index == _COMP_MAX_ROWS
    assert app.comp_scroll == 1


def test_up_scrolls_when_returning():
    """从末尾按 ↑ 回到可见区域时，comp_scroll 同步上移。"""
    n = _COMP_MAX_ROWS + 3
    app, handles = _make_comp_app(n)
    _handle_key(app, handles, Key.end)
    assert app.comp_scroll == n - _COMP_MAX_ROWS
    for _ in range(n - 1):
        _handle_key(app, handles, Key.up)
    assert app.comp_index == 0
    assert app.comp_scroll == 0


def test_highlight_always_visible():
    """循环导航时高亮项始终在可见窗口内。"""
    n = _COMP_MAX_ROWS + 5
    app, handles = _make_comp_app(n)
    visible = min(_COMP_MAX_ROWS, n)
    for _ in range(n * 2):
        _handle_key(app, handles, Key.down)
        assert app.comp_scroll <= app.comp_index < app.comp_scroll + visible


def test_highlight_in_window_rendered():
    """高亮项在滑动窗口内时正确显示。"""
    n = _COMP_MAX_ROWS + 3
    app, _ = _make_comp_app(n)
    app.comp_index = _COMP_MAX_ROWS
    app.comp_scroll = 1
    panel = _completion_panel(app, 80)
    plain = panel.renderable.plain
    assert f"/cmd{_COMP_MAX_ROWS}" in plain


# -- comp_scroll 重置 --


def test_scroll_reset_on_refresh():
    """refresh_completions 重置 comp_scroll。"""
    app = App(session_short_id="t")
    provider = make_provider([("a", "x"), ("b", "y")])
    app.input = "/"
    app.refresh_completions(provider)
    app.comp_scroll = 5
    app.refresh_completions(provider)
    assert app.comp_scroll == 0


def test_scroll_reset_on_dismiss():
    """dismiss_completions 重置 comp_scroll。"""
    app = App(session_short_id="t")
    app.comp_scroll = 3
    app.dismiss_completions()
    assert app.comp_scroll == 0


def test_scroll_reset_on_apply():
    """apply_completion 重置 comp_scroll。"""
    app = App(session_short_id="t")
    app.comp_scroll = 3
    app.apply_completion(CompletionItem(insert="/x", label="/x"))
    assert app.comp_scroll == 0


def test_scroll_reset_on_reset_input():
    """reset_input 重置 comp_scroll。"""
    app = App(session_short_id="t")
    app.comp_scroll = 3
    app.reset_input()
    assert app.comp_scroll == 0


# -- 每行截断 --


def test_long_label_truncated():
    """长 label 被截断到面板内容宽度，不导致多行。"""
    app = App(session_short_id="t")
    app.input = "/"
    app.comp_items = [CompletionItem(insert="/x", label="/x" + "A" * 100, hint="h")]
    app.comp_dismissed = False
    panel = _completion_panel(app, 40)
    assert panel.renderable.plain.count("\n") == 1


def test_long_hint_truncated():
    """长 hint 被截断，整行不超面板宽度。"""
    app = App(session_short_id="t")
    app.input = "/"
    app.comp_items = [CompletionItem(insert="/x", label="/x", hint="B" * 100)]
    app.comp_dismissed = False
    panel = _completion_panel(app, 40)
    assert panel.renderable.plain.count("\n") == 1


def test_wide_char_label_truncated():
    """宽字符 label 按显示宽度截断。"""
    app = App(session_short_id="t")
    app.input = "/"
    app.comp_items = [CompletionItem(insert="/x", label="中文" * 50, hint="h")]
    app.comp_dismissed = False
    panel = _completion_panel(app, 40)
    assert panel.renderable.plain.count("\n") == 1
