"""TUI status bar 渲染、context_window 与提示行测试。"""

from __future__ import annotations

import pytest
from mcc_core import PermissionRequest
from mcc_harness import DEFAULT_CONTEXT_WINDOW, context_window
from mcc_tui.app import App
from mcc_tui.completions import CompletionItem
from mcc_tui.render import _fmt_tokens, _status_hint, _status_info

# -- context_window --


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-4-20250514", 200_000),
        ("claude-sonnet-4-20250514", 200_000),
        ("claude-haiku-4-20250514", 200_000),
        ("gpt-4o", 128_000),
        ("deepseek-chat", 64_000),
        ("qwen-max", 128_000),
        ("some-unknown-model", DEFAULT_CONTEXT_WINDOW),
    ],
    ids=["opus4", "sonnet4", "haiku4", "gpt4", "deepseek", "qwen", "unknown"],
)
def test_context_window(model, expected):
    assert context_window(model) == expected


def test_context_window_default_is_200k():
    assert DEFAULT_CONTEXT_WINDOW == 200_000


# -- _fmt_tokens --


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "0"), (999, "999"), (1000, "1.0k"), (20000, "20.0k"), (200000, "200.0k")],
)
def test_fmt_tokens(n, expected):
    assert _fmt_tokens(n) == expected


# -- _status_info --


def _make_app(**kw) -> App:
    return App(session_short_id="t", **kw)


def _styles(app: App) -> list[str]:
    return [str(s.style) for s in _status_info(app).spans]


def test_status_info_shows_model_id():
    assert "claude-sonnet-4" in _status_info(_make_app(model_id="claude-sonnet-4")).plain


def test_status_info_no_model_placeholder():
    assert "—" in _status_info(_make_app()).plain


def test_status_info_ctx_ratio():
    app = _make_app(model_id="m", context_limit=200_000)
    app.ctx_tokens = 100_000
    assert "50%" in _status_info(app).plain


@pytest.mark.parametrize(
    ("ratio", "color", "near_limit"),
    [
        (0.5, "green", False),
        (0.6, "yellow", False),
        (0.84, "yellow", False),
        (0.85, "red", True),
        (0.95, "red", True),
    ],
    ids=["green", "yellow-low", "yellow-high", "red-threshold", "red-high"],
)
def test_status_info_color_grading(ratio, color, near_limit):
    app = _make_app(model_id="m", context_limit=100_000)
    app.ctx_tokens = int(100_000 * ratio)
    text = _status_info(app)
    assert any(color in s for s in _styles(app))
    assert ("near limit" in text.plain) == near_limit


def test_status_info_no_context_limit_shows_tokens_only():
    app = _make_app(model_id="m")
    app.ctx_tokens = 500
    text = _status_info(app)
    assert "ctx" in text.plain
    assert "500" in text.plain


def test_status_info_cache_placeholder():
    assert "cache" in _status_info(_make_app()).plain


def test_status_info_cache_ratio():
    app = _make_app()
    app.cache_seen_total = 200
    app.cache_read_total = 80
    assert "40%" in _status_info(app).plain


def test_status_info_cost():
    app = _make_app()
    app.cost_usd = 0.1234
    assert "$0.1234" in _status_info(app).plain


# -- _status_hint --


def test_status_hint_idle():
    assert "send" in _status_hint(_make_app()).plain


def test_status_hint_busy():
    app = _make_app()
    app.agent_busy = True
    assert "thinking" in _status_hint(app).plain


def test_status_hint_completion_active():
    app = _make_app()
    app.comp_items = [CompletionItem(insert="/h", label="/h")]
    text = _status_hint(app)
    assert "complete" in text.plain or "Tab" in text.plain


def test_status_hint_permission():
    app = _make_app()
    app.pending_permission = PermissionRequest(id=1, message="allow?")
    assert "allow" in _status_hint(app).plain


def test_status_hint_feedback_mode():
    app = _make_app()
    app.permission_feedback_mode = True
    assert "submit" in _status_hint(app).plain
