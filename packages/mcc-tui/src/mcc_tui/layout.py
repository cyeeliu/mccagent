"""TUI 布局引擎：统一计算各区域高度，确保不超终端。

压缩优先级（从先压缩到最后）：
1. 消息区域 — 可压缩到 0
2. 补全面板 — 可压缩到 0（隐藏）
3. 输入区域 — 可压缩到最小高度
4. 状态栏 / 选择面板 — 不可压缩（关键交互）
"""

from __future__ import annotations

from dataclasses import dataclass

_INPUT_ROWS = 3
_STATUS_ROWS = 3
_PANEL_BORDERS = 2
_TEXT_INDENT = 2
_MAX_INPUT_FRACTION = 4


@dataclass
class TerminalMetrics:
    width: int
    height: int


@dataclass
class LayoutResult:
    message_height: int
    input_height: int
    status_height: int
    comp_height: int
    sel_height: int
    text_width: int


class LayoutEngine:
    """根据终端尺寸和内容需求计算各区域高度。

    总高度 = (message_height + _PANEL_BORDERS) + comp + sel + input + status
    保证总高度 <= terminal.height。
    """

    def compute(
        self,
        terminal: TerminalMetrics,
        input_content_lines: int,
        comp_rows: int,
        sel_rows: int,
    ) -> LayoutResult:
        h = terminal.height
        w = terminal.width

        input_height = min(
            max(_INPUT_ROWS, input_content_lines + _PANEL_BORDERS),
            max(_INPUT_ROWS, h // _MAX_INPUT_FRACTION),
        )
        status_height = _STATUS_ROWS

        message_height = (
            h - _PANEL_BORDERS - comp_rows - sel_rows - input_height - status_height
        )

        if message_height < 1:
            if comp_rows > 0:
                saved = min(comp_rows, 1 - message_height)
                comp_rows -= saved
                message_height += saved

            if message_height < 1 and input_height > _INPUT_ROWS:
                saved = min(input_height - _INPUT_ROWS, 1 - message_height)
                input_height -= saved
                message_height += saved

            if message_height < 1 and sel_rows > 0:
                saved = min(sel_rows, 1 - message_height)
                sel_rows -= saved
                message_height += saved

            if message_height < 1 and status_height > 1:
                saved = min(status_height - 1, 1 - message_height)
                status_height -= saved
                message_height += saved

            if message_height < 1 and input_height > 1:
                saved = min(input_height - 1, 1 - message_height)
                input_height -= saved
                message_height += saved

        message_height = max(0, message_height)

        text_width = max(20, w - _PANEL_BORDERS - _TEXT_INDENT)

        return LayoutResult(
            message_height=message_height,
            input_height=input_height,
            status_height=status_height,
            comp_height=comp_rows,
            sel_height=sel_rows,
            text_width=text_width,
        )
