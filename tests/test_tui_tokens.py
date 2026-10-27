"""TUI token 统计、turn_stats 与流式估算测试。"""

from __future__ import annotations

import pytest
from mcc_core import LlmUsage, TextDelta, TurnEnd, UserEcho
from mcc_tui.app import App


def _usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
    iteration: int = 1,
) -> LlmUsage:
    return LlmUsage(
        iteration=iteration,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


# -- LlmUsage 累计 --


def test_llm_usage_accumulates_session():
    app = App(session_short_id="t")
    app.apply_event(_usage(input_tokens=100, output_tokens=50))
    app.apply_event(_usage(input_tokens=200, output_tokens=80, iteration=2))
    assert app.session_in == 300
    assert app.session_out == 130


def test_llm_usage_sets_ctx_tokens():
    app = App(session_short_id="t")
    app.apply_event(_usage(input_tokens=100, cache_read=30, cache_creation=20))
    assert app.ctx_tokens == 150
    assert app.turn_in == 150


def test_llm_usage_accumulates_cache():
    app = App(session_short_id="t")
    app.apply_event(_usage(input_tokens=100, cache_read=40))
    app.apply_event(_usage(input_tokens=100, cache_read=60, iteration=2))
    assert app.cache_read_total == 100
    # cache_seen_total 累加 turn_in（input + cache_read + cache_creation）
    assert app.cache_seen_total == 140 + 160


def test_cache_hit_pct_none_when_no_cache():
    assert App(session_short_id="t").cache_hit_pct is None


def test_cache_hit_pct_ratio():
    app = App(session_short_id="t")
    app.apply_event(_usage(input_tokens=60, cache_read=40))
    # turn_in = 60 + 40 = 100, hit = 40 / 100
    assert app.cache_hit_pct == 0.4


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("abcd", 1), ("hello world!", 3), ("a" * 100, 25)],
)
def test_est_stream_tokens(text, expected):
    app = App(session_short_id="t")
    app.streaming_text = text
    assert app.est_stream_tokens == expected


# -- turn_stats --


def test_turn_stats_emitted_on_turn_end():
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(UserEcho(text="hi"))
    app.apply_event(_usage(input_tokens=100, output_tokens=50))
    app.apply_event(TurnEnd(cost_usd=0.02))
    stats = [it for it in app.items if it.kind == "turn_stats"]
    assert len(stats) == 1
    assert "100" in stats[0].stats
    assert "50" in stats[0].stats
    assert "$0.0200" in stats[0].stats


def test_turn_stats_includes_cache_hit():
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(UserEcho(text="hi"))
    app.apply_event(_usage(input_tokens=100, output_tokens=50, cache_read=30))
    app.apply_event(TurnEnd(cost_usd=0.01))
    stats = [it for it in app.items if it.kind == "turn_stats"][0]
    assert "cache" in stats.stats
    assert "30" in stats.stats


def test_turn_stats_skipped_when_no_tokens():
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(TextDelta(text="resp"))
    app.apply_event(TurnEnd(cost_usd=0.0))
    assert not [it for it in app.items if it.kind == "turn_stats"]


def test_turn_stats_uses_run_delta_multi_turn():
    """多轮 LLM 调用：turn_stats 用 run 累计差值，非单轮值。"""
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(UserEcho(text="hi"))
    app.apply_event(_usage(input_tokens=100, output_tokens=50))
    app.apply_event(_usage(input_tokens=200, output_tokens=100, iteration=2))
    app.apply_event(TurnEnd(cost_usd=0.05))
    stats = [it for it in app.items if it.kind == "turn_stats"][0]
    assert "300" in stats.stats
    assert "150" in stats.stats


def test_turn_stats_reset_after_turn():
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(UserEcho(text="hi"))
    app.apply_event(_usage(input_tokens=100, output_tokens=50))
    app.apply_event(TurnEnd(cost_usd=0.01))
    assert app.turn_in == 0
    assert app.turn_out == 0


def test_user_echo_snapshots_run_start():
    app = App(session_short_id="t")
    app.apply_event(_usage(input_tokens=100, cache_read=50, output_tokens=50))
    # turn_in = 100 + 50 = 150, session_in = 150
    app.apply_event(UserEcho(text="second turn"))
    assert app._run_start_in == 150
    assert app._run_start_out == 50


def test_turn_stats_isolated_per_run():
    """第二个 run 的 turn_stats 只含该 run 的增量。"""
    app = App(session_short_id="t")
    app.agent_busy = True
    app.apply_event(UserEcho(text="run1"))
    app.apply_event(_usage(input_tokens=100, output_tokens=50))
    app.apply_event(TurnEnd(cost_usd=0.01))
    app.agent_busy = True
    app.apply_event(UserEcho(text="run2"))
    app.apply_event(_usage(input_tokens=200, output_tokens=80))
    app.apply_event(TurnEnd(cost_usd=0.02))
    stats = [it for it in app.items if it.kind == "turn_stats"]
    assert len(stats) == 2
    assert "200" in stats[1].stats
    assert "80" in stats[1].stats


# -- CLI 注入 --


async def test_cli_injects_tui_handles(monkeypatch):
    from mcc_cli import _run_tui_mode
    from mcc_cli.slash import build_default_registry

    captured: dict = {}

    async def fake_run_tui(handles, console=None):
        captured["handles"] = handles

    import mcc_tui

    monkeypatch.setattr(mcc_tui, "run_tui", fake_run_tui)

    class FakeAgent:
        model = "claude-opus-4-1"

    class FakeSource:
        agent = FakeAgent()
        slash_registry = build_default_registry()

        async def run(self, user_input):
            if False:
                yield

    await _run_tui_mode(FakeSource())

    h = captured["handles"]
    assert h.model_id == "claude-opus-4-1"
    assert h.context_limit == 200_000
    assert h.completions is not None
    items = h.completions("/he")
    assert any(i.insert == "/help" for i in items)
