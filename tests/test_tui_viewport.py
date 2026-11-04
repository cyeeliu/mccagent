"""InputViewport 与输入框视口测试（UI-03）。

验证光标始终可见、水平/垂直滚动正确、多行输入支持、
长文本粘贴不导致布局错乱、CJK/Emoji 混排正确处理。
"""

from __future__ import annotations

import pytest

from mcc_tui.metrics import TerminalTextMetrics as metrics
from mcc_tui.viewport import InputViewport


# ────────────────────────────── fixtures ──────────────────────────────

@pytest.fixture
def vp() -> InputViewport:
    return InputViewport()


# ────────────────────────────── _cursor_to_line_col ──────────────────────────────

class TestCursorToLineCol:
    def test_single_line(self) -> None:
        assert InputViewport._cursor_to_line_col("hello", 3) == (0, 3)

    def test_multiline(self) -> None:
        text = "line1\nline2\nline3"
        assert InputViewport._cursor_to_line_col(text, 0) == (0, 0)
        assert InputViewport._cursor_to_line_col(text, 5) == (0, 5)
        assert InputViewport._cursor_to_line_col(text, 6) == (1, 0)
        assert InputViewport._cursor_to_line_col(text, 11) == (1, 5)
        assert InputViewport._cursor_to_line_col(text, 12) == (2, 0)

    def test_cursor_at_end(self) -> None:
        text = "ab\ncd"
        assert InputViewport._cursor_to_line_col(text, len(text)) == (1, 2)

    def test_empty(self) -> None:
        assert InputViewport._cursor_to_line_col("", 0) == (0, 0)


# ────────────────────────────── _horizontal_slice ──────────────────────────────

class TestHorizontalSlice:
    def test_no_scroll(self) -> None:
        assert InputViewport._horizontal_slice("hello", 0, 10) == "hello"

    def test_scroll_past(self) -> None:
        assert InputViewport._horizontal_slice("hello", 5, 10) == ""

    def test_partial(self) -> None:
        assert InputViewport._horizontal_slice("hello", 2, 3) == "llo"

    def test_chinese(self) -> None:
        assert InputViewport._horizontal_slice("你好世界", 2, 4) == "好世"

    def test_chinese_half_scroll(self) -> None:
        result = InputViewport._horizontal_slice("你好世界", 1, 4)
        assert metrics.width(result) <= 4

    def test_empty(self) -> None:
        assert InputViewport._horizontal_slice("", 0, 10) == ""

    def test_width_zero(self) -> None:
        assert InputViewport._horizontal_slice("hello", 0, 0) == ""


# ────────────────────────────── update ──────────────────────────────

class TestUpdate:
    def test_cursor_visible_no_scroll(self, vp: InputViewport) -> None:
        vp.update("hello", 2, 80, 1)
        assert vp.scroll_x == 0
        assert vp.scroll_y == 0

    def test_cursor_right_scroll(self, vp: InputViewport) -> None:
        text = "a" * 100
        vp.update(text, 95, 20, 1)
        assert vp.scroll_x > 0
        vis_row, vis_col = vp.cursor_visible_pos(text, 95)
        assert 0 <= vis_col < 20

    def test_cursor_left_scroll(self, vp: InputViewport) -> None:
        vp.scroll_x = 50
        vp.update("hello", 0, 80, 1)
        assert vp.scroll_x == 0

    def test_vertical_scroll_down(self, vp: InputViewport) -> None:
        text = "\n".join(f"line{i}" for i in range(20))
        vp.update(text, len(text), 80, 3)
        assert vp.scroll_y > 0
        vis_row, _ = vp.cursor_visible_pos(text, len(text))
        assert 0 <= vis_row < 3

    def test_vertical_scroll_up(self, vp: InputViewport) -> None:
        vp.scroll_y = 10
        text = "line0\nline1\nline2"
        vp.update(text, 0, 80, 3)
        assert vp.scroll_y == 0

    def test_chinese_cursor_scroll(self, vp: InputViewport) -> None:
        text = "你" * 50
        vp.update(text, 45, 20, 1)
        vis_row, vis_col = vp.cursor_visible_pos(text, 45)
        assert 0 <= vis_col < 20

    def test_zero_width(self, vp: InputViewport) -> None:
        vp.update("hello", 2, 0, 1)
        assert vp.scroll_x == 0
        assert vp.scroll_y == 0

    def test_zero_height(self, vp: InputViewport) -> None:
        vp.update("hello", 2, 80, 0)
        assert vp.scroll_x == 0
        assert vp.scroll_y == 0


# ────────────────────────────── visible_lines ──────────────────────────────

class TestVisibleLines:
    def test_short_text(self, vp: InputViewport) -> None:
        vp.update("hello", 0, 80, 1)
        assert vp.visible_lines("hello", 80, 1) == ["hello"]

    def test_long_text_horizontal(self, vp: InputViewport) -> None:
        text = "a" * 100
        vp.update(text, 99, 20, 1)
        lines = vp.visible_lines(text, 20, 1)
        assert len(lines) == 1
        assert metrics.width(lines[0]) <= 20

    def test_multiline(self, vp: InputViewport) -> None:
        text = "line0\nline1\nline2\nline3"
        vp.update(text, 0, 80, 2)
        lines = vp.visible_lines(text, 80, 2)
        assert len(lines) == 2
        assert lines[0] == "line0"
        assert lines[1] == "line1"

    def test_multiline_scrolled(self, vp: InputViewport) -> None:
        text = "\n".join(f"line{i}" for i in range(10))
        vp.update(text, len(text), 80, 3)
        lines = vp.visible_lines(text, 80, 3)
        assert len(lines) == 3

    def test_pads_to_height(self, vp: InputViewport) -> None:
        vp.update("hi", 0, 80, 5)
        lines = vp.visible_lines("hi", 80, 5)
        assert len(lines) == 5
        assert lines[0] == "hi"
        assert lines[1] == ""

    def test_chinese_visible(self, vp: InputViewport) -> None:
        text = "你好世界你好世界"
        vp.update(text, 0, 6, 1)
        lines = vp.visible_lines(text, 6, 1)
        assert metrics.width(lines[0]) <= 6


# ────────────────────────────── cursor_visible_pos ──────────────────────────────

class TestCursorVisiblePos:
    def test_no_scroll(self, vp: InputViewport) -> None:
        vp.update("hello", 2, 80, 1)
        row, col = vp.cursor_visible_pos("hello", 2)
        assert row == 0
        assert col == 2

    def test_after_scroll(self, vp: InputViewport) -> None:
        text = "a" * 100
        vp.update(text, 50, 20, 1)
        row, col = vp.cursor_visible_pos(text, 50)
        assert row == 0
        assert 0 <= col < 20

    def test_multiline(self, vp: InputViewport) -> None:
        text = "line0\nline1\nline2"
        vp.update(text, 8, 80, 3)
        row, col = vp.cursor_visible_pos(text, 8)
        assert row == 1
        assert col == 2

    def test_chinese_width(self, vp: InputViewport) -> None:
        text = "你好世界"
        vp.update(text, 2, 80, 1)
        row, col = vp.cursor_visible_pos(text, 2)
        assert row == 0
        assert col == 4


# ────────────────────────────── reset ──────────────────────────────

class TestReset:
    def test_reset_clears_scroll(self, vp: InputViewport) -> None:
        vp.scroll_x = 10
        vp.scroll_y = 5
        vp.reset()
        assert vp.scroll_x == 0
        assert vp.scroll_y == 0


# ────────────────────────────── integration: cursor always visible ──────────────────────────────

class TestCursorAlwaysVisible:
    @pytest.mark.parametrize(
        "text,cursor,width,height",
        [
            ("a" * 200, 150, 20, 1),
            ("a" * 200, 0, 20, 1),
            ("a" * 200, 199, 20, 1),
            ("\n".join(f"line{i}" for i in range(50)), 200, 80, 3),
            ("你好" * 50, 80, 20, 1),
            ("Hello 世界 🌍 " * 20, 100, 30, 1),
        ],
    )
    def test_cursor_in_viewport(
        self,
        text: str,
        cursor: int,
        width: int,
        height: int,
    ) -> None:
        vp = InputViewport()
        vp.update(text, cursor, width, height)
        vis_row, vis_col = vp.cursor_visible_pos(text, cursor)
        assert 0 <= vis_row < height
        assert vis_col >= 0

    def test_cursor_movement_tracking(self) -> None:
        """模拟光标左右移动，视口始终跟随。"""
        vp = InputViewport()
        text = "a" * 100
        width = 20

        vp.update(text, 0, width, 1)
        assert vp.scroll_x == 0

        for cursor in range(0, 100, 5):
            vp.update(text, cursor, width, 1)
            _, vis_col = vp.cursor_visible_pos(text, cursor)
            assert 0 <= vis_col < width

        for cursor in range(99, -1, -5):
            vp.update(text, cursor, width, 1)
            _, vis_col = vp.cursor_visible_pos(text, cursor)
            assert 0 <= vis_col < width

    def test_long_text_paste(self) -> None:
        """长文本粘贴后光标在末尾仍可见。"""
        vp = InputViewport()
        text = "粘贴的很长文本内容" * 30
        cursor = len(text)
        vp.update(text, cursor, 30, 1)
        vis_row, vis_col = vp.cursor_visible_pos(text, cursor)
        assert 0 <= vis_col < 30

    def test_multiline_edit(self) -> None:
        """多行文本编辑时光标行始终可见。"""
        vp = InputViewport()
        text = "\n".join(f"第{i}行" for i in range(30))
        for cursor in [0, len(text) // 2, len(text)]:
            vp.update(text, cursor, 80, 5)
            vis_row, _ = vp.cursor_visible_pos(text, cursor)
            assert 0 <= vis_row < 5
