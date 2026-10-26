"""Goal Loop 测试：状态机、evaluator、transcript、AgentLoop 集成。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_core import Message, TextBlock, ToolContext, Usage
from mcc_core.event import GoalStatus, Notice, TurnEnd
from mcc_harness import (
    AgentLoopBuilder,
    GoalController,
    GoalError,
    PromptGoalEvaluator,
    transcript_text,
)
from mcc_llm import MessageStop, TextDelta
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session", depth=0)


@pytest.fixture
def registry() -> object:
    return default_registry()


def _make_agent(llm, ctx, registry, **kwargs):
    builder = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5")
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


async def _collect(queue: asyncio.Queue) -> list:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def _eval_turn(ok: bool, reason: str, impossible: bool = False) -> list:
    """构造 evaluator MockLlm 的一轮：返回 JSON 判断。"""
    payload = json.dumps({"ok": ok, "reason": reason, "impossible": impossible})
    return [TextDelta(text=payload)]


# ---------- transcript_text ----------


def test_transcript_text_basic():
    msgs = [
        Message.user("hello"),
        Message.assistant("hi there"),
    ]
    text = transcript_text(msgs)
    assert "USER:" in text
    assert "hello" in text
    assert "ASSISTANT:" in text
    assert "hi there" in text


def test_transcript_text_truncation():
    """超大最新消息做头尾截断。"""
    big = "x" * 50000
    msgs = [Message.user(big)]
    text = transcript_text(msgs, max_characters=1000)
    assert len(text) <= 1100
    assert "...[middle omitted]..." in text


# ---------- GoalController 状态机 ----------


def _controller(eval_llm: MockLlm, **kwargs) -> GoalController:
    evaluator = PromptGoalEvaluator(llm=eval_llm, model="claude-haiku-4-5")
    return GoalController(evaluator=evaluator, **kwargs)


def test_set_goal_and_status():
    ctrl = _controller(MockLlm([]))
    ctrl.set_goal("pytest passes")
    assert ctrl.active is not None
    assert ctrl.active.condition == "pytest passes"
    status = ctrl.status()
    assert "Goal active" in status
    assert "pytest passes" in status


def test_set_goal_empty_raises():
    ctrl = _controller(MockLlm([]))
    with pytest.raises(GoalError, match="empty"):
        ctrl.set_goal("")


def test_set_goal_too_long_raises():
    ctrl = _controller(MockLlm([]), max_goal_length=10)
    with pytest.raises(GoalError, match="exceed"):
        ctrl.set_goal("x" * 11)


def test_clear_goal():
    ctrl = _controller(MockLlm([]))
    ctrl.set_goal("pytest passes")
    result = ctrl.clear()
    assert "cleared" in result
    assert ctrl.active is None


def test_clear_no_goal():
    ctrl = _controller(MockLlm([]))
    assert ctrl.clear() == "No goal set"


@pytest.mark.asyncio
async def test_evaluate_no_goal_allows():
    ctrl = _controller(MockLlm([]))
    decision = await ctrl.evaluate_after_turn([])
    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_evaluate_achieved():
    eval_llm = MockLlm([_eval_turn(ok=True, reason="pytest passed")])
    ctrl = _controller(eval_llm)
    ctrl.set_goal("pytest passes")
    decision = await ctrl.evaluate_after_turn([Message.user("done")])
    assert decision.action == "achieved"
    assert ctrl.active is None


@pytest.mark.asyncio
async def test_evaluate_block_then_achieved():
    eval_llm = MockLlm([
        _eval_turn(ok=False, reason="pytest not run"),
        _eval_turn(ok=True, reason="pytest passed"),
    ])
    ctrl = _controller(eval_llm)
    ctrl.set_goal("pytest passes")
    d1 = await ctrl.evaluate_after_turn([Message.user("maybe")])
    assert d1.action == "block"
    assert ctrl.active is not None
    d2 = await ctrl.evaluate_after_turn([Message.user("now done")])
    assert d2.action == "achieved"
    assert ctrl.active is None


@pytest.mark.asyncio
async def test_evaluate_impossible():
    eval_llm = MockLlm([_eval_turn(ok=False, reason="no such file", impossible=True)])
    ctrl = _controller(eval_llm)
    ctrl.set_goal("fix nonexistent")
    decision = await ctrl.evaluate_after_turn([Message.user("can't")])
    assert decision.action == "failed"
    assert ctrl.active is None


@pytest.mark.asyncio
async def test_evaluate_block_cap_limit():
    turns = [_eval_turn(ok=False, reason="not yet") for _ in range(10)]
    eval_llm = MockLlm(turns)
    ctrl = _controller(eval_llm, block_cap=3)
    ctrl.set_goal("never done")
    actions = []
    for _ in range(5):
        d = await ctrl.evaluate_after_turn([Message.user("working")])
        actions.append(d.action)
    assert "block" in actions
    assert "limit" in actions


# ---------- AgentLoop 集成 ----------


@pytest.mark.asyncio
async def test_goal_block_continues_then_achieved(ctx, registry):
    """模型想停 → evaluator block → 追加消息继续 → evaluator achieved → 返回。"""
    worker = MockLlm([
        single_text_turn("I think I'm done."),
        single_text_turn("Now pytest passes, really done."),
    ])
    evaluator_llm = MockLlm([
        _eval_turn(ok=False, reason="pytest not run yet"),
        _eval_turn(ok=True, reason="pytest passed"),
    ])
    evaluator = PromptGoalEvaluator(llm=evaluator_llm, model="claude-haiku-4-5")
    goal = GoalController(evaluator=evaluator)
    agent = _make_agent(worker, ctx, registry, goal=goal)
    goal.set_goal("pytest passes")

    queue: asyncio.Queue = asyncio.Queue()
    run = await agent.run_streaming("fix the bug", queue)
    events = await _collect(queue)

    assert run.iterations == 2
    statuses = [e for e in events if isinstance(e, GoalStatus)]
    assert any(s.action == "block" for s in statuses)
    assert any(s.action == "achieved" for s in statuses)


@pytest.mark.asyncio
async def test_no_goal_behavior_unchanged(ctx, registry):
    """无 goal 时行为与历史版本一致：模型 end_turn 即返回。"""
    llm = MockLlm([single_text_turn("hello")])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()
    run = await agent.run_streaming("hi", queue)
    assert run.iterations == 1
    assert run.final_text == "hello"


@pytest.mark.asyncio
async def test_goal_failed_stops(ctx, registry):
    """evaluator 判断 impossible → 返回，不继续循环。"""
    worker = MockLlm([single_text_turn("can't do it")])
    evaluator_llm = MockLlm([
        _eval_turn(ok=False, reason="no such file", impossible=True),
    ])
    evaluator = PromptGoalEvaluator(llm=evaluator_llm, model="claude-haiku-4-5")
    goal = GoalController(evaluator=evaluator)
    agent = _make_agent(worker, ctx, registry, goal=goal)
    goal.set_goal("fix nonexistent")

    queue: asyncio.Queue = asyncio.Queue()
    run = await agent.run_streaming("try", queue)
    events = await _collect(queue)
    assert run.iterations == 1
    assert any(
        isinstance(e, GoalStatus) and e.action == "failed" for e in events
    )
