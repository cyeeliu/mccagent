"""TUI 输入补全过滤与浮层状态机测试。"""

from __future__ import annotations

import asyncio

import pytest
from mcc_tui import TuiHandles, _handle_key
from mcc_tui.app import App
from mcc_tui.completions import (
    CompletionItem,
    filter_completions,
    make_provider,
)
from mcc_tui.keys import Key


@pytest.fixture
def items() -> list[CompletionItem]:
    return [
        CompletionItem(insert="/help", label="/help", hint="show help"),
        CompletionItem(insert="/model", label="/model", hint="switch model"),
        CompletionItem(insert="/clear", label="/clear", hint="clear history"),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/h", 1),
        ("/", 3),
        ("/x", 0),
        ("help", 0),
        ("/help arg", 0),
        ("/M", 1),
    ],
    ids=["prefix-h", "all", "no-match", "not-slash", "has-space", "case-insensitive"],
)
def test_filter_completions(items, text, expected):
    assert len(filter_completions(items, text)) == expected


def test_filter_completions_case_insensitive(items):
    result = filter_completions(items, "/MODEL")
    assert len(result) == 1
    assert result[0].insert == "/model"


def test_completion_item_frozen(items):
    with pytest.raises(AttributeError):
        items[0].insert = "/x"


def test_make_provider_commands():
    provider = make_provider([("help", "show help"), ("model", "switch")])
    result = provider("/")
    assert {r.insert for r in result} == {"/help", "/model"}
    assert all(r.hint for r in result)


def test_make_provider_with_skills():
    provider = make_provider([("help", "show")], skill_names=["review", "test"])
    result = provider("/")
    assert {r.insert for r in result} == {"/help", "/review", "/test"}
    skill = next(r for r in result if r.insert == "/review")
    assert skill.hint == "skill"


def test_make_provider_empty():
    assert make_provider([])("/") == []


# -- App 浮层状态机 --


def test_comp_active_initially_false():
    assert not App(session_short_id="t").comp_active


def test_refresh_completions_populates():
    app = App(session_short_id="t")
    provider = make_provider([("help", "h"), ("model", "m")])
    app.input = "/h"
    app.refresh_completions(provider)
    assert app.comp_active
    assert len(app.comp_items) == 1


def test_refresh_completions_none_provider_clears():
    app = App(session_short_id="t")
    app.comp_items = [CompletionItem(insert="/x", label="/x")]
    app.refresh_completions(None)
    assert app.comp_items == []


def test_refresh_completions_dismissed_clears():
    app = App(session_short_id="t")
    provider = make_provider([("help", "h")])
    app.dismiss_completions()
    app.input = "/h"
    app.refresh_completions(provider)
    assert not app.comp_active


def test_dismiss_then_reset_reopens():
    app = App(session_short_id="t")
    provider = make_provider([("help", "h")])
    app.input = "/h"
    app.refresh_completions(provider)
    assert app.comp_active
    app.dismiss_completions()
    assert not app.comp_active
    app.reset_input()
    app.input = "/h"
    app.refresh_completions(provider)
    assert app.comp_active


def test_apply_completion_sets_input():
    app = App(session_short_id="t")
    app.apply_completion(CompletionItem(insert="/help", label="/help"))
    assert app.input == "/help"
    assert app.cursor == len("/help")
    assert app.comp_items == []


def test_reset_input_clears_all():
    app = App(session_short_id="t")
    app.input = "/he"
    app.cursor = 3
    app.comp_dismissed = True
    app.comp_items = [CompletionItem(insert="/x", label="/x")]
    app.reset_input()
    assert app.input == ""
    assert app.cursor == 0
    assert app.comp_items == []
    assert not app.comp_dismissed


def test_refresh_completions_index_clamped():
    app = App(session_short_id="t")
    provider = make_provider([("a", "x"), ("b", "y"), ("c", "z")])
    app.comp_index = 5
    app.input = "/"
    app.refresh_completions(provider)
    assert app.comp_index <= len(app.comp_items) - 1


# -- 键位交互 --


def _key_handles(provider=None):
    return TuiHandles(
        events=asyncio.Queue(),
        input_tx=asyncio.Queue(),
        session_short_id="t",
        completions=provider,
    )


def _key_provider():
    return make_provider([("help", "h"), ("status", "s"), ("model", "m")])


def test_key_slash_triggers_completions():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    assert app.input == "/"
    assert app.comp_active
    assert len(app.comp_items) == 3


def test_key_down_navigates():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, Key.down)
    assert app.comp_index == 1


def test_key_up_wraps():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, Key.up)
    assert app.comp_index == len(app.comp_items) - 1


def test_key_tab_completes():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, Key.tab)
    assert app.input == "/help"
    assert not app.comp_active


def test_key_escape_closes_panel():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, Key.escape)
    assert not app.comp_active


def test_key_enter_exact_match_sends():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    for ch in "help":
        _handle_key(app, h, ch)
    assert app.input == "/help"
    assert app.comp_active
    _handle_key(app, h, Key.enter)
    assert app.input == ""
    assert not h.input_tx.empty()


def test_key_enter_non_exact_completes_not_send():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, "h")
    assert app.input == "/h"
    assert app.comp_active
    _handle_key(app, h, Key.enter)
    assert app.input == "/help"
    assert h.input_tx.empty()


def test_key_backspace_refreshes():
    app = App(session_short_id="t")
    h = _key_handles(_key_provider())
    _handle_key(app, h, "/")
    _handle_key(app, h, "h")
    assert len(app.comp_items) == 1
    _handle_key(app, h, Key.backspace)
    assert app.input == "/"
    assert len(app.comp_items) == 3
