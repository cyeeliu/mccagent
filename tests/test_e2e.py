"""端到端测试：完整 Agent 流程（用户输入 → LLM → 工具 → 最终输出）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_cli.events import AgentEventSource
from mcc_cli.headless import run_headless
from mcc_core import ToolContext, Usage
from mcc_core.event import (
    Error as ErrorEvent,
)
from mcc_core.event import (
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_harness import AgentLoopBuilder, PermissionChecker
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="e2e", depth=0)


@pytest.fixture
def registry():
    return default_registry()


def _agent(llm, ctx, registry, **kwargs):
    builder = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5")
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


async def _drain(source, prompt) -> list:
    events = []
    async for ev in source.run(prompt):
        events.append(ev)
    return events


def _event_types(events) -> list[str]:
    return [type(e).__name__ for e in events]


# ---------- E2E: 纯文本对话 ----------


@pytest.mark.asyncio
async def test_e2e_text_only(ctx, registry):
    """用户提问 → LLM 直接回答（无工具调用）。"""
    llm = MockLlm([single_text_turn("Paris is the capital of France.")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "What is the capital of France?")

    types = _event_types(events)
    assert types[0] == "UserEcho"
    assert "TextDelta" in types
    assert types[-1] == "TurnEnd"

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "Paris" in text


# ---------- E2E: 工具调用循环 ----------


@pytest.mark.asyncio
async def test_e2e_write_then_read(ctx, registry, tmp_path):
    """LLM 调用 write_file 写文件 → 再调 read_file 读回 → 返回内容。"""
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(
            partial_json=json.dumps({"path": "greeting.txt", "content": "hello world"})
        ),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = [
        ToolUseStart(id="t2", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "greeting.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn3 = single_text_turn("The file contains: hello world")
    llm = MockLlm([turn1, turn2, turn3])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Write and read a file")

    assert (tmp_path / "greeting.txt").read_text() == "hello world"
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(tool_starts) == 2
    assert tool_starts[0].name == "write_file"
    assert tool_starts[1].name == "read_file"

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert all(not e.is_error for e in tool_ends)

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "hello world" in text


# ---------- E2E: bash 工具 ----------


@pytest.mark.asyncio
async def test_e2e_bash_echo(ctx, registry):
    """LLM 调用 run_bash 执行 echo 命令。"""
    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "echo e2e-success"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Command output: e2e-success")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Run echo")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 1
    assert "e2e-success" in tool_ends[0].output


# ---------- E2E: 权限拒绝 ----------


@pytest.mark.asyncio
async def test_e2e_permission_deny(ctx, registry):
    """deny 规则阻止 rm 命令，工具返回 permission denied。"""
    from mcc_config import PermissionConfig

    cfg = PermissionConfig(deny=["Bash(rm:*)"])
    checker = PermissionChecker(cfg)

    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "rm -rf /"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Permission was denied.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry, permission=checker))
    events = await _drain(source, "Delete everything")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert tool_ends[0].is_error
    assert "permission denied" in tool_ends[0].output


# ---------- E2E: headless 模式 ----------


@pytest.mark.asyncio
async def test_e2e_headless_mode(ctx, registry, capsys):
    """通过 run_headless 运行，stdout 输出最终答案。"""
    llm = MockLlm([single_text_turn("42")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    exit_code = await run_headless(source, "What is 6x7?", quiet=False)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "42" in captured.out


@pytest.mark.asyncio
async def test_e2e_headless_quiet_mode(ctx, registry, capsys):
    """quiet 模式下 stderr 不输出工具进度。"""
    llm = MockLlm([single_text_turn("silent answer")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    exit_code = await run_headless(source, "test", quiet=True)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "silent answer" in captured.out
    assert captured.err == ""


# ---------- E2E: 成本追踪 ----------


@pytest.mark.asyncio
async def test_e2e_cost_tracking(ctx, registry, capsys):
    """headless 模式 stderr 输出 USD 成本。"""
    from mcc_core import Usage

    llm = MockLlm([single_text_turn("done", Usage(input_tokens=500, output_tokens=200))])
    source = AgentEventSource(_agent(llm, ctx, registry))
    await run_headless(source, "test", quiet=False)
    captured = capsys.readouterr()

    assert "$" in captured.err
    assert "USD" in captured.err


# ---------- E2E: 多轮工具调用 ----------


@pytest.mark.asyncio
async def test_e2e_multi_tool_parallel(ctx, registry, tmp_path):
    """LLM 同时请求多个工具调用，并行执行。"""
    from mcc_core import Usage

    u = Usage()
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "a.txt", "content": "aaa"})),
        ToolUseStart(id="t2", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "b.txt", "content": "bbb"})),
        MessageStop(stop_reason="tool_use", usage=u),
    ]
    turn2 = single_text_turn("Both files written.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Write two files")

    assert (tmp_path / "a.txt").read_text() == "aaa"
    assert (tmp_path / "b.txt").read_text() == "bbb"
    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 2
    assert all(not e.is_error for e in tool_ends)


# ---------- E2E: 错误传播 ----------


@pytest.mark.asyncio
async def test_e2e_tool_error_continues(ctx, registry):
    """工具失败不中断循环，LLM 继续根据错误结果回复。"""
    from mcc_core import Usage

    turn1 = [
        ToolUseStart(id="t1", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "nonexistent.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("The file does not exist.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Read a missing file")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert tool_ends[0].is_error
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "does not exist" in text


# ---------- E2E: 事件完整性 ----------


@pytest.mark.asyncio
async def test_e2e_event_sequence(ctx, registry):
    """验证事件序列：UserEcho → [TextDelta|ToolCall*] → TurnEnd。"""
    llm = MockLlm([single_text_turn("hello")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "hi")

    assert isinstance(events[0], UserEcho)
    assert isinstance(events[-1], TurnEnd)
    for ev in events[1:-1]:
        assert isinstance(ev, (TextDelta, ToolCallStart, ToolCallEnd, Notice))
