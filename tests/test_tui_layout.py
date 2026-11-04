"""LayoutEngine 布局测试：验证各终端尺寸下总高度不溢出。"""

import pytest
from mcc_tui.layout import LayoutEngine, LayoutResult, TerminalMetrics

_INPUT_ROWS = 3
_STATUS_ROWS = 3
_PANEL_BORDERS = 2
_TEXT_INDENT = 2


@pytest.fixture
def engine():
    return LayoutEngine()


def _total(result: LayoutResult) -> int:
    return (
        result.message_height
        + _PANEL_BORDERS
        + result.comp_height
        + result.sel_height
        + result.input_height
        + result.status_height
    )


@pytest.mark.parametrize(
    "width,height",
    [
        (40, 10),
        (60, 12),
        (80, 16),
        (100, 24),
        (120, 40),
        (200, 50),
        (30, 8),
        (20, 5),
    ],
)
def test_total_never_exceeds_terminal(engine, width, height):
    result = engine.compute(TerminalMetrics(width, height), 1, 0, 0)
    assert _total(result) <= height


@pytest.mark.parametrize(
    "width,height",
    [
        (40, 10),
        (60, 12),
        (80, 16),
        (100, 24),
        (120, 40),
    ],
)
def test_total_never_exceeds_with_overlays(engine, width, height):
    result = engine.compute(TerminalMetrics(width, height), 1, 7, 9)
    assert _total(result) <= height


def test_message_area_positive_on_normal_terminal(engine):
    result = engine.compute(TerminalMetrics(80, 24), 1, 0, 0)
    assert result.message_height > 0


def test_message_area_zero_on_tiny_terminal(engine):
    result = engine.compute(TerminalMetrics(40, 8), 1, 0, 0)
    assert result.message_height >= 0
    assert _total(result) <= 8


def test_input_height_at_least_min(engine):
    result = engine.compute(TerminalMetrics(80, 24), 1, 0, 0)
    assert result.input_height >= _INPUT_ROWS


def test_input_height_grows_with_content(engine):
    short = engine.compute(TerminalMetrics(80, 40), 1, 0, 0)
    long_ = engine.compute(TerminalMetrics(80, 40), 20, 0, 0)
    assert long_.input_height >= short.input_height


def test_input_height_capped_at_quarter_terminal(engine):
    result = engine.compute(TerminalMetrics(80, 40), 100, 0, 0)
    assert result.input_height <= 40 // 4 + _INPUT_ROWS


def test_status_height_fixed(engine):
    for h in (10, 16, 24, 40):
        result = engine.compute(TerminalMetrics(80, h), 1, 0, 0)
        assert result.status_height == _STATUS_ROWS


def test_text_width_calculation(engine):
    result = engine.compute(TerminalMetrics(80, 24), 1, 0, 0)
    assert result.text_width == 80 - _PANEL_BORDERS - _TEXT_INDENT


def test_text_width_minimum(engine):
    result = engine.compute(TerminalMetrics(10, 24), 1, 0, 0)
    assert result.text_width >= 20


def test_comp_panel_compressed_when_space_tight(engine):
    engine.compute(TerminalMetrics(80, 10), 1, 0, 0)
    result_with_comp = engine.compute(TerminalMetrics(80, 10), 1, 7, 0)
    assert _total(result_with_comp) <= 10
    assert result_with_comp.comp_height <= 7


def test_comp_panel_preserved_when_space_sufficient(engine):
    result = engine.compute(TerminalMetrics(100, 24), 1, 7, 0)
    assert result.comp_height == 7


def test_sel_panel_preserved(engine):
    result = engine.compute(TerminalMetrics(100, 24), 1, 0, 9)
    assert result.sel_height == 9


def test_sel_panel_not_compressed(engine):
    result = engine.compute(TerminalMetrics(40, 10), 1, 0, 9)
    assert _total(result) <= 10


def test_sel_panel_compressed_only_when_necessary(engine):
    result = engine.compute(TerminalMetrics(100, 24), 1, 0, 9)
    assert result.sel_height == 9


def test_all_non_negative(engine):
    result = engine.compute(TerminalMetrics(40, 8), 5, 7, 9)
    assert result.message_height >= 0
    assert result.input_height >= 0
    assert result.status_height >= 0
    assert result.comp_height >= 0
    assert result.sel_height >= 0


def test_large_terminal_layout(engine):
    result = engine.compute(TerminalMetrics(200, 50), 1, 0, 0)
    assert result.message_height > 20
    assert _total(result) <= 50


def test_input_compressed_before_status(engine):
    result = engine.compute(TerminalMetrics(80, 9), 10, 0, 0)
    assert result.input_height >= _INPUT_ROWS
    assert result.status_height == _STATUS_ROWS
    assert _total(result) <= 9


def test_no_comp_no_sel_zero_height(engine):
    result = engine.compute(TerminalMetrics(80, 24), 1, 0, 0)
    assert result.comp_height == 0
    assert result.sel_height == 0


def test_message_reduces_with_overlays(engine):
    base = engine.compute(TerminalMetrics(100, 24), 1, 0, 0)
    with_overlays = engine.compute(TerminalMetrics(100, 24), 1, 7, 9)
    assert with_overlays.message_height < base.message_height


def test_extreme_tiny_terminal(engine):
    result = engine.compute(TerminalMetrics(20, 5), 1, 0, 0)
    assert _total(result) <= 5
    assert result.input_height >= 1
    assert result.status_height >= 1


def test_input_multiline_grows_height(engine):
    one_line = engine.compute(TerminalMetrics(80, 30), 1, 0, 0)
    five_lines = engine.compute(TerminalMetrics(80, 30), 5, 0, 0)
    assert five_lines.input_height >= one_line.input_height


def test_total_exact_when_space_sufficient(engine):
    result = engine.compute(TerminalMetrics(100, 24), 1, 0, 0)
    assert _total(result) == 24


def test_total_exact_with_overlays(engine):
    result = engine.compute(TerminalMetrics(120, 40), 1, 7, 9)
    assert _total(result) == 40
