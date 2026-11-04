"""TerminalTextMetrics 与 render 宽度计算测试（UI-02）。

验证显示宽度（cell width）计算覆盖 ASCII / 中文 / 全角 / Emoji 混排，
以及 render 层 _truncate / _wrap_line / _split_lines 统一使用显示宽度。
"""

from __future__ import annotations

import pytest
from rich.text import Text

from mcc_tui.metrics import TerminalTextMetrics as metrics
from mcc_tui.render import _split_lines, _truncate, _wrap_line


# ────────────────────────────── fixtures ──────────────────────────────

@pytest.fixture(
    params=[
        "Hello World",
        "你好，世界",
        "AgentOS 架构设计",
        "Hello 世界 🌍",
        "中文 Emoji 🚀 混排",
    ]
)
def sample_text(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=[40, 60, 80, 120])
def term_width(request: pytest.FixtureRequest) -> int:
    return request.param


# ────────────────────────────── width ──────────────────────────────

class TestWidth:
    def test_ascii(self) -> None:
        assert metrics.width("Hello World") == 11

    def test_chinese(self) -> None:
        assert metrics.width("你好") == 4

    def test_fullwidth_punct(self) -> None:
        assert metrics.width("，") == 2
        assert metrics.width("！") == 2

    def test_emoji(self) -> None:
        assert metrics.width("🌍") == 2
        assert metrics.width("🚀") == 2

    def test_mixed(self) -> None:
        assert metrics.width("Hello 世界 🌍") == 13

    def test_empty(self) -> None:
        assert metrics.width("") == 0

    def test_control_char_fallback(self) -> None:
        assert metrics.width("\x00") == 0
        assert metrics.width("a\x00b") >= 2

    @pytest.mark.parametrize(
        "text",
        ["Hello World", "你好，世界", "AgentOS 架构设计", "Hello 世界 🌍", "中文 Emoji 🚀 混排"],
    )
    def test_width_non_negative(self, text: str) -> None:
        assert metrics.width(text) >= 0


# ────────────────────────────── truncate ──────────────────────────────

class TestTruncate:
    def test_short_no_truncate(self) -> None:
        assert _truncate("Hello", 80) == "Hello"

    def test_ascii_truncate(self) -> None:
        result = _truncate("Hello World", 8)
        assert metrics.width(result) <= 8
        assert result.endswith("…")

    def test_chinese_truncate(self) -> None:
        result = _truncate("你好，世界", 5)
        assert metrics.width(result) <= 5
        assert result.endswith("…")

    def test_emoji_truncate(self) -> None:
        result = _truncate("🌍🌍🌍🌍", 5)
        assert metrics.width(result) <= 5

    def test_exact_width_no_truncate(self) -> None:
        assert _truncate("你好", 4) == "你好"

    def test_zero_width(self) -> None:
        assert _truncate("Hello", 0) == ""

    def test_width_one_no_ellipsis_overflow(self) -> None:
        result = _truncate("Hello", 1)
        assert metrics.width(result) <= 1

    def test_truncate_preserves_width_limit(
        self, sample_text: str, term_width: int
    ) -> None:
        result = _truncate(sample_text, term_width)
        assert metrics.width(result) <= term_width


# ────────────────────────────── wrap ──────────────────────────────

class TestWrap:
    def test_short_no_wrap(self) -> None:
        assert metrics.wrap("Hello", 80) == ["Hello"]

    def test_ascii_wrap(self) -> None:
        result = metrics.wrap("Hello World", 5)
        assert all(metrics.width(line) <= 5 for line in result)
        assert "".join(result) == "Hello World"

    def test_chinese_wrap(self) -> None:
        result = metrics.wrap("你好世界", 6)
        assert all(metrics.width(line) <= 6 for line in result)
        assert "".join(result) == "你好世界"

    def test_mixed_wrap(self) -> None:
        text = "Hello 世界 🌍"
        result = metrics.wrap(text, 7)
        assert all(metrics.width(line) <= 7 for line in result)
        assert "".join(result) == text

    def test_empty(self) -> None:
        assert metrics.wrap("", 80) == [""]

    def test_wrap_preserves_content(
        self, sample_text: str, term_width: int
    ) -> None:
        result = metrics.wrap(sample_text, term_width)
        assert "".join(result) == sample_text

    def test_wrap_respects_width_limit(
        self, sample_text: str, term_width: int
    ) -> None:
        result = metrics.wrap(sample_text, term_width)
        for line in result:
            assert metrics.width(line) <= term_width


# ────────────────────────────── wrap_offsets ──────────────────────────────

class TestWrapOffsets:
    def test_short2short_offsets(self) -> None:
        assert metrics.wrap_offsets("Hello", 80) == [(0, 5)]

    def test_offsets_match_wrap(self) -> None:
        text = "你好世界Hello"
        width = 6
        offsets = metrics.wrap_offsets(text, width)
        wrapped = metrics.wrap(text, width)
        assert len(offsets) == len(wrapped)
        for (s, e), line in zip(offsets, wrapped, strict=True):
            assert text[s:e] == line

    def test_offsets_cover_full_text(self) -> None:
        text = "Hello 世界 🌍 你好"
        offsets = metrics.wrap_offsets(text, 5)
        assert offsets[0][0] == 0
        assert offsets[-1][1] == len(text)
        for i in range(len(offsets) - 1):
            assert offsets[i][1] == offsets[i + 1][0]

    def test_empty(self) -> None:
        assert metrics.wrap_offsets("", 80) == [(0, 0)]


# ────────────────────────────── render integration ──────────────────────────────

class TestRenderWidthIntegration:
    def test_truncate_uses_display_width(self) -> None:
        result = _truncate("你好世界", 5)
        assert metrics.width(result) <= 5
        assert result.endswith("…")

    def test_wrap_line_uses_display_width(self) -> None:
        text = Text("你好世界你好世界")
        result = _wrap_line(text, 0, len(text.plain), 6)
        for line in result:
            assert metrics.width(line.plain) <= 6

    def test_wrap_line_preserves_content(self) -> None:
        text = Text("你好世界Hello")
        result = _wrap_line(text, 0, len(text.plain), 5)
        joined = "".join(line.plain for line in result)
        assert joined == "你好世界Hello"

    def test_split_lines_uses_display_width(self) -> None:
        text = Text("你好世界你好世界\nHello World")
        result = _split_lines(text, 6)
        for line in result:
            assert metrics.width(line.plain) <= 6

    def test_split_lines_multiline_chinese(self) -> None:
        text = Text("你好\n世界\n测试")
        result = _split_lines(text, 80)
        assert len(result) == 3
        assert result[0].plain == "你好"
        assert result[1].plain == "世界"
        assert result[2].plain == "测试"

    def test_wrap_line_preserves_style(self) -> None:
        text = Text()
        text.append("你好世界你好世界", style="bold red")
        result = _wrap_line(text, 0, len(text.plain), 6)
        assert len(result) > 1
        for line in result:
            assert any(span.style == "bold red" for span in line.spans)

    def test_no_overflow_mixed(
        self, sample_text: str, term_width: int
    ) -> None:
        text = Text(sample_text)
        result = _split_lines(text, term_width)
        for line in result:
            assert metrics.width(line.plain) <= term_width
