"""终端文本度量：基于 wcwidth 的显示宽度计算。

统一所有布局计算使用的 cell width 度量，避免用 Python 字符数
（len()）近似终端显示宽度——中文字符、全角标点、Emoji 等的
显示宽度与字符数不一致，会导致换行错位和截断异常。
"""

from __future__ import annotations

from wcwidth import wcswidth

_ELLIPSIS = "…"
_ELLIPSIS_WIDTH = max(0, wcswidth(_ELLIPSIS))


class TerminalTextMetrics:
    """终端 cell width 统一度量。"""

    @staticmethod
    def width(text: str) -> int:
        """返回 *text* 在终端中占用的显示宽度（cell 数）。

        wcswidth 对包含不可打印控制字符的字符串返回 -1，
        此处兜底为 0 以避免负宽度参与布局计算。
        """
        if not text:
            return 0
        return max(0, wcswidth(text))

    @staticmethod
    def char_width(ch: str) -> int:
        """返回单个字符的显示宽度。"""
        if not ch:
            return 0
        return max(0, wcswidth(ch))

    @staticmethod
    def truncate(text: str, max_width: int) -> str:
        """截断 *text* 使其显示宽度不超过 *max_width*，末尾加省略号。

        若 *max_width* 连省略号都放不下，则按宽度逐字符截取不加省略号。
        """
        if max_width <= 0:
            return ""
        if TerminalTextMetrics.width(text) <= max_width:
            return text
        target = max_width - _ELLIPSIS_WIDTH
        if target <= 0:
            result: list[str] = []
            current = 0
            for ch in text:
                cw = TerminalTextMetrics.char_width(ch)
                if current + cw > max_width:
                    break
                result.append(ch)
                current += cw
            return "".join(result)
        result = []
        current = 0
        for ch in text:
            cw = TerminalTextMetrics.char_width(ch)
            if current + cw > target:
                break
            result.append(ch)
            current += cw
        return "".join(result) + _ELLIPSIS

    @staticmethod
    def wrap(text: str, max_width: int) -> list[str]:
        """按显示宽度 *max_width* 对 *text* 换行，返回行列表。

        逐字符累计显示宽度，超出时断行。不做词级拆分，
        与原有 _wrap_line 行为一致。
        """
        if max_width <= 0:
            return [text] if text else [""]
        if not text:
            return [""]
        result: list[str] = []
        current_line: list[str] = []
        current_w = 0
        for ch in text:
            cw = TerminalTextMetrics.char_width(ch)
            if current_w + cw > max_width and current_line:
                result.append("".join(current_line))
                current_line = []
                current_w = 0
            current_line.append(ch)
            current_w += cw
        result.append("".join(current_line))
        return result

    @staticmethod
    def wrap_offsets(text: str, max_width: int) -> list[tuple[int, int]]:
        """按显示宽度换行，返回每行的 (start, end) 字符偏移列表。

        供需要保留 Rich Text span 样式的调用方使用：
        偏移基于原始字符位置，可直接用于 _extract_line。
        """
        if not text:
            return [(0, 0)]
        if max_width <= 0:
            return [(0, len(text))]
        result: list[tuple[int, int]] = []
        chunk_start = 0
        current_w = 0
        for i, ch in enumerate(text):
            cw = TerminalTextMetrics.char_width(ch)
            if current_w + cw > max_width and chunk_start < i:
                result.append((chunk_start, i))
                chunk_start = i
                current_w = 0
            current_w += cw
        result.append((chunk_start, len(text)))
        return result
