"""AgentLoop 主循环测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_config import PermissionConfig
from mcc_core import (
    AgentError,
    TextBlock,
    ToolContext,
    ToolUseBlock,
    Usage,
)
from mcc_core.event import (
    LlmUsage,
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_harness import (
    AgentLoop,
    AgentLoopBuilder,
    AgentRun,
    PermissionChecker,
    permission_request_for,
)
from mcc_harness.permission import BashAction, Deny, PathAction, PermissionRequest
from mcc_llm import MessageStop, StreamEvent, ToolUseInputDelta, ToolUseStart
from mcc_llm import TextDelta as LlmTextDelta
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry, default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-session", depth=0)


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


def _make_agent(
    llm: MockLlm,
    ctx: ToolContext,
    registry: ToolRegistry,
    **kwargs,
) -> AgentLoop:
    builder = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5-20251001")
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


async def _collect_events(queue: asyncio.Queue) -> list:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


# ---------- 基本流式运行 ----------


@pytest.mark.asyncio
async def test_streaming_end_turn_events(ctx, registry):
    llm = MockLlm([single_text_turn("Hello, world!")])
    agent = _make_agent(llm, ctx, registry, system="You are a test agent.")
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("hi", queue)
    events = await _collect_events(queue)

    assert run.final_text == "Hello, world!"
    assert run.iterations == 1
    assert any(isinstance(e, UserEcho) for e in events)
    assert any(isinstance(e, TextDelta) for e in events)
    assert any(isinstance(e, TurnEnd) for e in events)


@pytest.mark.asyncio
async def test_headless_run(ctx, registry):
    llm = MockLlm([single_text_turn("42")])
    agent = _make_agent(llm, ctx, registry)
    run = await agent.run("what is 6x7?")
    assert run.final_text == "42"


@pytest.mark.asyncio
async def test_headless_run_returns_usage(ctx, registry):
    usage = Usage(input_tokens=100, output_tokens=50)
    llm = MockLlm([single_text_turn("done", usage)])
    agent = _make_agent(llm, ctx, registry)
    run = await agent.run("test")
    assert run.total_usage.input_tokens == 100
    assert run.total_usage.output_tokens == 50


# ---------- 成本计算 ----------


@pytest.mark.asyncio
async def test_cost_calculation_nonzero(ctx, registry):
    usage = Usage(input_tokens=10, output_tokens=5)
    llm = MockLlm([single_text_turn("done", usage)])
    agent = _make_agent(llm, ctx, registry)
    run = await agent.run("test")
    assert run.cost_usd > 0.0


@pytest.mark.asyncio
async def test_cost_haiku_pricing(ctx, registry):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    llm = MockLlm([single_text_turn("done", usage)])
    agent = _make_agent(llm, ctx, registry)
    run = await agent.run("test")
    assert run.cost_usd == pytest.approx(0.80 + 4.0)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("claude-opus-4-8", (15.0, 75.0)),
        ("claude-sonnet-4-6", (3.0, 15.0)),
        ("claude-haiku-4-5-20251001", (0.80, 4.0)),
        ("gpt-4o", (2.5, 10.0)),
        ("gpt-4-turbo", (10.0, 30.0)),
        ("gpt-3.5-turbo", (0.5, 1.5)),
        ("unknown-model", (3.0, 15.0)),
    ],
)
def test_model_price_per_1m(model: str, expected: tuple[float, float]):
    from mcc_harness import model_price_per_1m

    assert model_price_per_1m(model) == expected


# ---------- Budget 限制 ----------


@pytest.mark.asyncio
async def test_budget_limit_raises(ctx, registry):
    """超出预算限制时抛出 AgentError。"""
    usage = Usage(input_tokens=10_000_000, output_tokens=10_000_000)
    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "echo hi"})),
        MessageStop(stop_reason="tool_use", usage=usage),
    ]
    turn2 = single_text_turn("done")
    llm = MockLlm([turn1, turn2])
    agent = _make_agent(llm, ctx, registry, max_cost_usd=0.01)
    with pytest.raises(AgentError, match="cost"):
        await agent.run("test")


@pytest.mark.asyncio
async def test_max_iterations_raises(ctx, registry):
    """超出最大迭代次数时抛出 AgentError。"""
    def tool_turn(i: int) -> list[StreamEvent]:
        return [
            ToolUseStart(id=f"t{i}", name="run_bash"),
            ToolUseInputDelta(partial_json=json.dumps({"command": "echo hi"})),
            MessageStop(stop_reason="tool_use", usage=Usage(input_tokens=1, output_tokens=1)),
        ]

    turns = [tool_turn(i) for i in range(50)]
    llm = MockLlm(turns)
    agent = _make_agent(llm, ctx, registry, max_iterations=3)
    with pytest.raises(AgentError, match="max_iterations"):
        await agent.run("test")


# ---------- 工具调用循环 ----------


@pytest.mark.asyncio
async def test_tool_use_cycle(ctx, registry, tmp_path):
    """LLM 请求工具调用 → 工具执行 → LLM 返回最终文本。"""
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "out.txt", "content": "hi"})),
        MessageStop(stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=5)),
    ]
    turn2 = single_text_turn("File written successfully.")
    llm = MockLlm([turn1, turn2])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("write a file", queue)
    events = await _collect_events(queue)

    assert run.final_text == "File written successfully."
    assert run.iterations == 2
    assert (tmp_path / "out.txt").read_text() == "hi"
    assert any(isinstance(e, ToolCallStart) and e.name == "write_file" for e in events)
    assert any(isinstance(e, ToolCallEnd) and not e.is_error for e in events)


@pytest.mark.asyncio
async def test_llm_usage_emitted_per_turn(ctx, registry):
    """每轮 LLM 调用完成后发射 LlmUsage 事件（token 统计权威来源）。"""
    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "ls"})),
        MessageStop(stop_reason="tool_use",
                    usage=Usage(input_tokens=1520, output_tokens=310)),
    ]
    turn2 = [
        TextDelta(text="done"),
        MessageStop(stop_reason="end_turn",
                    usage=Usage(input_tokens=2000, output_tokens=90)),
    ]
    llm = MockLlm([turn1, turn2])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()

    await agent.run_streaming("run ls", queue)
    events = await _collect_events(queue)

    usages = [e for e in events if isinstance(e, LlmUsage)]
    assert len(usages) == 2
    assert usages[0].input_tokens == 1520
    assert usages[0].output_tokens == 310
    assert usages[0].iteration == 1  # 主循环 iteration 从 1 起
    assert usages[0].cost_usd > 0
    assert usages[1].input_tokens == 2000
    assert usages[1].iteration == 2
    # 事件时序：一轮的 usage 在工具执行（ToolCallEnd）之前
    idx_usage0 = events.index(usages[0])
    idx_tool_end = next(
        i for i, e in enumerate(events) if isinstance(e, ToolCallEnd)
    )
    assert idx_usage0 < idx_tool_end


@pytest.mark.asyncio
async def test_tool_call_start_carries_full_args(ctx, registry):
    """ToolCallStart 携带完整参数 JSON（参数分片流完后发射）。

    下游（评测轨迹的 args 断言、TUI 提示）依赖事件里的参数；
    空 args_preview 会让 args_regex 断言恒失效。
    """
    args_json = json.dumps({"path": "out.txt", "content": "hi"})
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json=args_json),
        MessageStop(stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=5)),
    ]
    llm = MockLlm([turn1, single_text_turn("done.")])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()

    await agent.run_streaming("write a file", queue)
    events = await _collect_events(queue)

    starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(starts) == 1
    assert starts[0].args_preview == args_json
    # 参数流完成、工具执行前发射：先 tool_call 后 tool_result 的顺序不变
    idx_call = events.index(starts[0])
    idx_end = next(i for i, e in enumerate(events) if isinstance(e, ToolCallEnd))
    assert idx_call < idx_end


@pytest.mark.asyncio
async def test_tool_error_propagates_as_tool_result(ctx, registry):
    """工具失败时产生 is_error=True 的 ToolResult，不中断循环。"""
    turn1 = [
        ToolUseStart(id="t1", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "nonexistent.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("File not found.")
    llm = MockLlm([turn1, turn2])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("read missing file", queue)
    events = await _collect_events(queue)

    assert run.final_text == "File not found."
    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 1
    assert tool_ends[0].is_error


# ---------- 权限集成 ----------


@pytest.mark.asyncio
async def test_permission_deny_blocks_tool(ctx, registry):
    """deny 规则阻止工具执行，返回 permission denied 错误。"""
    cfg = PermissionConfig(mode=None, allow=[], deny=["Bash(rm:*)"])
    checker = PermissionChecker(cfg)

    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "rm -rf /"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Denied.")
    llm = MockLlm([turn1, turn2])

    builder = AgentLoopBuilder(llm, registry, ctx).permission(checker)
    agent = builder.build()
    queue: asyncio.Queue = asyncio.Queue()

    await agent.run_streaming("rm everything", queue)
    events = await _collect_events(queue)

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 1
    assert tool_ends[0].is_error
    assert "permission denied" in tool_ends[0].output


# ---------- permission_request_for 映射 ----------


@pytest.mark.parametrize(
    "tool_name,input,expected_category,expected_action_type",
    [
        ("read_file", {"path": "src/main.py"}, "Read", "path"),
        ("list_dir", {"path": "."}, "Read", "path"),
        ("grep", {"pattern": "x"}, "Read", "path"),
        ("glob", {"pattern": "*.py"}, "Read", "path"),
        ("write_file", {"path": "out.txt"}, "Write", "path"),
        ("edit_file", {"path": "out.txt"}, "Edit", "path"),
        ("run_bash", {"command": "ls"}, "Bash", "bash"),
        ("unknown_tool", {}, "Bash", "bash"),
    ],
)
def test_permission_request_mapping(
    tool_name: str,
    input: dict,
    expected_category: str,
    expected_action_type: str,
):
    req = permission_request_for(tool_name, input)
    assert req.category == expected_category
    if expected_action_type == "path":
        assert isinstance(req.action, PathAction)
    else:
        assert isinstance(req.action, BashAction)


def test_permission_request_bash_uses_real_command():
    """run_bash 映射必须传递真实 command，不是工具名。"""
    req = permission_request_for("run_bash", {"command": "rm -rf /"})
    assert isinstance(req.action, BashAction)
    assert req.action.cmd == "rm -rf /"


def test_permission_request_deny_matches_real_command():
    """deny 规则 Bash(rm:*) 能匹配 run_bash 的真实 command。"""
    cfg = PermissionConfig(mode=None, allow=[], deny=["Bash(rm:*)"])
    checker = PermissionChecker(cfg)
    req = permission_request_for("run_bash", {"command": "rm -rf /"})
    assert isinstance(checker.check(req), Deny)


# ---------- 事件顺序 ----------


@pytest.mark.asyncio
async def test_event_order_user_echo_first(ctx, registry):
    llm = MockLlm([single_text_turn("hi")])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("hello", queue)
    events = await _collect_events(queue)
    assert isinstance(events[0], UserEcho)
    assert events[0].text == "hello"


@pytest.mark.asyncio
async def test_event_order_turn_end_last(ctx, registry):
    llm = MockLlm([single_text_turn("done")])
    agent = _make_agent(llm, ctx, registry)
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    events = await _collect_events(queue)
    assert isinstance(events[-1], TurnEnd)


# ---------- 重试逻辑 ----------


@pytest.mark.asyncio
async def test_retry_on_429(ctx, registry):
    """429 错误触发重试，最终成功。"""
    from mcc_llm import CompleteRequest, MessageResponse

    class RetryLlm:
        def __init__(self):
            self._calls = 0

        async def complete(self, req: CompleteRequest) -> MessageResponse:
            pass

        async def stream(self, req: CompleteRequest):
            self._calls += 1
            if self._calls < 3:
                raise RuntimeError("429 Too Many Requests")
            yield LlmTextDelta(text="recovered")
            yield MessageStop(stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=3))

    llm = RetryLlm()
    agent = _make_agent(llm, ctx, registry)
    run = await agent.run("test")
    assert run.final_text == "recovered"
    assert llm._calls == 3


@pytest.mark.asyncio
async def test_retry_exhausted_raises(ctx, registry):
    from mcc_llm import CompleteRequest

    class Always429:
        async def complete(self, req: CompleteRequest):
            pass

        async def stream(self, req: CompleteRequest):
            raise RuntimeError("429 Too Many Requests")
            yield  # makes this an async generator

    llm = Always429()
    agent = _make_agent(llm, ctx, registry)
    with pytest.raises(RuntimeError, match="429"):
        await agent.run("test")
