"""输入框视口管理：确保光标始终可见，支持水平/垂直滚动。

当输入文本超出可见区域时，自动调整滚动偏移使光标保持可见。
基于 TerminalTextMetrics 的显示宽度计算，正确处理 CJK 和 Emoji。
"""

from __future__ import annotations

from dataclasses import dataclass

from mcc_tui.metrics import TerminalTextMetrics as _metrics


@dataclass
class InputViewport:
    """输入框视口：跟踪水平/垂直滚动偏移，确保光标始终可见。"""

    scroll_x: int = 0
    scroll_y: int = 0

    def update(self, text: str, cursor: int, width: int, height: int) -> None:
        """根据文本、光标和可用尺寸调整视口，确保光标可见。"""
        if width <= 0 or height <= 0:
            self.scroll_x = 0
            self.scroll_y = 0
            return

        lines = text.split("\n")
        line_no, col = self._cursor_to_line_col(text, cursor)

        if line_no < self.scroll_y:
            self.scroll_y = line_no
        elif line_no >= self.scroll_y + height:
            self.scroll_y = line_no - height + 1
        max_sy = max(0, len(lines) - height)
        self.scroll_y = max(0, min(self.scroll_y, max_sy))

        current_line = lines[line_no] if line_no < len(lines) else ""
        col_width = _metrics.width(current_line[:col])
        if col_width < self.scroll_x:
            self.scroll_x = col_width
        elif col_width >= self.scroll_x + width:
            self.scroll_x = col_width - width + 1
        max_sx = max(0, _metrics.width(current_line) - width + 1)
        self.scroll_x = max(0, min(self.scroll_x, max_sx))

    def visible_lines(self, text: str, width: int, height: int) -> list[str]:
        """返回可见行列表，每行按水平偏移截取。"""
        lines = text.split("\n")
        result: list[str] = []
        end = min(self.scroll_y + height, len(lines))
        for i in range(self.scroll_y, end):
            result.append(self._horizontal_slice(lines[i], self.scroll_x, width))
        while len(result) < height:
            result.append("")
        return result

    def cursor_visible_pos(
        self, text: str, cursor: int
    ) -> tuple[int, int]:
        """返回光标在可见区域中的 (行, 显示列) 位置。"""
        line_no, col = self._cursor_to_line_col(text, cursor)
        vis_row = line_no - self.scroll_y
        lines = text.split("\n")
        current_line = lines[line_no] if line_no < len(lines) else ""
        col_width = _metrics.width(current_line[:col])
        vis_col = col_width - self.scroll_x
        return vis_row, vis_col

    def reset(self) -> None:
        self.scroll_x = 0
        self.scroll_y = 0

    @staticmethod
    def _cursor_to_line_col(text: str, cursor: int) -> tuple[int, int]:
        """将字符偏移 cursor 转换为 (行号, 列号)。"""
        line_no = 0
        col = 0
        for i, ch in enumerate(text):
            if i >= cursor:
                break
            if ch == "\n":
                line_no += 1
                col = 0
            else:
                col += 1
        return line_no, col

    @staticmethod
    def _horizontal_slice(line: str, scroll_x: int, width: int) -> str:
        """按显示宽度从 scroll_x 偏移开始截取 width 宽度的文本。"""
        if not line or width <= 0:
            return ""
        result: list[str] = []
        current_w = 0
        visible_w = 0
        for ch in line:
            ch_w = _metrics.char_width(ch)
            char_start = current_w
            current_w += ch_w
            if char_start < scroll_x:
                continue
            if visible_w + ch_w > width:
                break
            result.append(ch)
            visible_w += ch_w
        return "".join(result)
