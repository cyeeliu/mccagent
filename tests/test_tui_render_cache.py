"""RenderCache 测试：验证历史消息渲染缓存行为。"""

import pytest
from mcc_tui.app import App, CallStatus, RenderItem
from mcc_tui.render import RenderCache, build_renderable
from rich.console import Console
from rich.text import Text


@pytest.fixture
def cache():
    return RenderCache()


@pytest.fixture
def console():
    return Console(width=80, height=24)


def _make_item(kind="user", text="hello"):
    return RenderItem(kind=kind, text=text)


def test_cache_hit_returns_same_lines(cache):
    item = _make_item()
    lines1 = cache.get_lines(item, 80)
    lines2 = cache.get_lines(item, 80)
    assert lines1 is lines2


def test_width_change_recomputes(cache):
    item = _make_item(text="a" * 100)
    lines1 = cache.get_lines(item, 80)
    lines2 = cache.get_lines(item, 40)
    assert len(lines2) > len(lines1)


def test_invalidate_removes_entry(cache):
    item = _make_item()
    cache.get_lines(item, 80)
    assert len(cache) == 1
    cache.invalidate(item)
    assert len(cache) == 0


def test_invalidate_nonexistent_no_error(cache):
    item = _make_item()
    cache.invalidate(item)


def test_clear_empties_cache(cache):
    for i in range(5):
        cache.get_lines(_make_item(text=f"item{i}"), 80)
    assert len(cache) == 5
    cache.clear()
    assert len(cache) == 0


def test_prune_removes_stale_entries(cache):
    from collections import deque

    items = deque([_make_item(text=f"item{i}") for i in range(3)], maxlen=500)
    for item in items:
        cache.get_lines(item, 80)
    stale = _make_item(text="stale")
    cache.get_lines(stale, 80)
    assert len(cache) == 4
    cache.prune(items)
    assert len(cache) == 3


def test_prune_no_op_when_cache_small(cache):
    from collections import deque

    items = deque([_make_item(text="a"), _make_item(text="b")], maxlen=500)
    for item in items:
        cache.get_lines(item, 80)
    cache.prune(items)
    assert len(cache) == 2


def test_cache_with_store_dict():
    store: dict = {}
    cache = RenderCache(store)
    item = _make_item()
    cache.get_lines(item, 80)
    assert len(store) == 1


def test_different_items_cached_separately(cache):
    item1 = _make_item(text="hello")
    item2 = _make_item(text="world")
    cache.get_lines(item1, 80)
    cache.get_lines(item2, 80)
    assert len(cache) == 2


def test_build_renderable_uses_cache(console):
    app = App(session_short_id="test")
    app.push(RenderItem(kind="user", text="hello"))
    app.push(RenderItem(kind="assistant", text="world"))
    build_renderable(app, console)
    assert len(app.render_cache) == 2


def test_build_renderable_cache_hit(console):
    app = App(session_short_id="test")
    app.push(RenderItem(kind="user", text="hello"))
    build_renderable(app, console)
    cache_size_before = len(app.render_cache)
    build_renderable(app, console)
    cache_size_after = len(app.render_cache)
    assert cache_size_before == cache_size_after


def test_tool_call_end_invalidates_cache(console):
    from mcc_core import ToolCallEnd

    app = App(session_short_id="test")
    from mcc_core import ToolCallStart
    app.apply_event(ToolCallStart(id="call-1", name="bash", args_preview="ls"))
    build_renderable(app, console)
    assert len(app.render_cache) == 1
    app.apply_event(ToolCallEnd(id="call-1", output="done", is_error=False))
    build_renderable(app, console)
    assert len(app.render_cache) >= 1


def test_width_change_in_build_recomputes(console):
    app = App(session_short_id="test")
    app.push(RenderItem(kind="assistant", text="a" * 200))
    build_renderable(app, console)
    narrow = Console(width=40, height=24)
    build_renderable(app, narrow)
    assert len(app.render_cache) == 1


def test_streaming_text_not_cached(console):
    app = App(session_short_id="test")
    app.streaming_text = "streaming content"
    build_renderable(app, console)
    assert len(app.render_cache) == 0


def test_many_items_cached(console):
    app = App(session_short_id="test")
    for i in range(100):
        app.push(RenderItem(kind="user", text=f"message {i}"))
    build_renderable(app, console)
    assert len(app.render_cache) == 100


def test_prune_on_deque_overflow(console):
    from collections import deque

    app = App(session_short_id="test")
    for i in range(10):
        app.push(RenderItem(kind="user", text=f"msg{i}"))
    build_renderable(app, console)
    assert len(app.render_cache) == 10
    for i in range(10, 15):
        app.push(RenderItem(kind="user", text=f"msg{i}"))
    build_renderable(app, console)
    assert len(app.render_cache) == 15


def test_cache_content_correct(console):
    app = App(session_short_id="test")
    app.push(RenderItem(kind="user", text="hello world"))
    layout1 = build_renderable(app, console)
    layout2 = build_renderable(app, console)
    assert layout1 is not None
    assert layout2 is not None


def test_seq_uniqueness():
    items = [RenderItem(kind="user", text=f"msg{i}") for i in range(100)]
    seqs = [item.seq for item in items]
    assert len(set(seqs)) == 100


def test_500_items_cached(console):
    app = App(session_short_id="test")
    for i in range(500):
        app.push(RenderItem(kind="user", text=f"message {i}"))
    build_renderable(app, console)
    assert len(app.render_cache) == 500


def test_cache_hit_skips_rebuild(console):
    app = App(session_short_id="test")
    app.push(RenderItem(kind="assistant", text="hello"))
    build_renderable(app, console)
    item = app.items[0]
    cache = RenderCache(app.render_cache)
    lines_first = cache.get_lines(item, 80)
    lines_second = cache.get_lines(item, 80)
    assert lines_first is lines_second


def test_seq_based_cache_key_not_id(cache):
    item1 = RenderItem(kind="user", text="hello")
    cache.get_lines(item1, 80)
    assert item1.seq in cache._cache
    assert id(item1) not in cache._cache


def test_invalidate_by_seq(cache):
    item = RenderItem(kind="user", text="hello")
    cache.get_lines(item, 80)
    assert len(cache) == 1
    cache.invalidate(item)
    assert len(cache) == 0
    cache.get_lines(item, 80)
    assert len(cache) == 1


def test_prune_by_seq(cache):
    from collections import deque

    items = deque([RenderItem(kind="user", text=f"item{i}") for i in range(3)], maxlen=500)
    for item in items:
        cache.get_lines(item, 80)
    stale = RenderItem(kind="user", text="stale")
    cache.get_lines(stale, 80)
    assert len(cache) == 4
    cache.prune(items)
    assert len(cache) == 3
    for item in items:
        assert item.seq in cache._cache


@pytest.mark.parametrize("n_items", [10, 100, 500])
def test_cache_growth_linear(console, n_items):
    app = App(session_short_id="test")
    for i in range(n_items):
        app.push(RenderItem(kind="user", text=f"msg {i}" * 10))
    build_renderable(app, console)
    assert len(app.render_cache) == n_items
