"""MockLlm 编程式使用示例。

展示如何在不依赖真实 API key 的情况下，用 MockLlm 驱动 AgentLoop
完成各种场景的端到端测试。

用法（在项目根目录）：
    uv run python examples/run_mock.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcc_core import ToolContext, Usage
from mcc_core.event import Notice, TextDelta, ToolCallEnd, ToolCallStart, TurnEnd, UserEcho
from mcc_harness import AgentLoopBuilder, PermissionChecker
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry


def _agent(llm, ctx, registry, **kwargs):
    builder = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5")
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


async def _drain(agent, prompt) -> list:
    from mcc_cli.events import AgentEventSource

    events = []
    async for ev in AgentEventSource(agent).run(prompt):
        events.append(ev)
    return events


def _print_events(events) -> None:
    for ev in events:
        if isinstance(ev, UserEcho):
            print(f"  [UserEcho] {ev.text}")
        elif isinstance(ev, TextDelta):
            print(f"  [TextDelta] {ev.text}", end="")
        elif isinstance(ev, ToolCallStart):
            print(f"\n  [ToolCallStart] {ev.name}")
        elif isinstance(ev, ToolCallEnd):
            status = "ERROR" if ev.is_error else "OK"
            print(f"  [ToolCallEnd] {status}: {ev.output[:80]}")
        elif isinstance(ev, Notice):
            print(f"  [Notice] {ev.text}")
        elif isinstance(ev, TurnEnd):
            print(f"\n  [TurnEnd] cost=${ev.cost_usd:.6f}")


async def demo_simple_text(tmp: Path) -> None:
    print("\n=== Demo 1: 纯文本对话 ===")
    ctx = ToolContext(cwd=tmp, session_id="demo1")
    llm = MockLlm([single_text_turn("Paris is the capital of France.")])
    agent = _agent(llm, ctx, default_registry())
    events = await _drain(agent, "What is the capital of France?")
    _print_events(events)


async def demo_write_then_read(tmp: Path) -> None:
    print("\n=== Demo 2: 写文件 → 读文件 ===")
    ctx = ToolContext(cwd=tmp, session_id="demo2")
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(
            partial_json=json.dumps({"path": "hello.txt", "content": "hello from mock"})
        ),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = [
        ToolUseStart(id="t2", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "hello.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn3 = single_text_turn('File content is "hello from mock".')
    llm = MockLlm([turn1, turn2, turn3])
    agent = _agent(llm, ctx, default_registry())
    events = await _drain(agent, "Write and read hello.txt")
    _print_events(events)
    print(f"  [File] {tmp / 'hello.txt'} = {(tmp / 'hello.txt').read_text()}")


async def demo_parallel_tools(tmp: Path) -> None:
    print("\n=== Demo 3: 并行工具调用 ===")
    ctx = ToolContext(cwd=tmp, session_id="demo3")
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "a.txt", "content": "AAA"})),
        ToolUseStart(id="t2", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "b.txt", "content": "BBB"})),
        ToolUseStart(id="t3", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "c.txt", "content": "CCC"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("All three files written in parallel.")
    llm = MockLlm([turn1, turn2])
    agent = _agent(llm, ctx, default_registry())
    events = await _drain(agent, "Write three files at once")
    _print_events(events)
    for f in ("a.txt", "b.txt", "c.txt"):
        print(f"  [File] {f} = {(tmp / f).read_text()}")


async def demo_permission_deny(tmp: Path) -> None:
    print("\n=== Demo 4: 权限拒绝 ===")
    from mcc_config import PermissionConfig

    ctx = ToolContext(cwd=tmp, session_id="demo4")
    checker = PermissionChecker(PermissionConfig(deny=["Bash(rm:*)"]))
    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "rm -rf /"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Permission was denied for rm command.")
    llm = MockLlm([turn1, turn2])
    agent = _agent(llm, ctx, default_registry(), permission=checker)
    events = await _drain(agent, "Delete everything")
    _print_events(events)


async def demo_cost_tracking(tmp: Path) -> None:
    print("\n=== Demo 5: 成本追踪 ===")
    ctx = ToolContext(cwd=tmp, session_id="demo5")
    usage = Usage(input_tokens=5000, output_tokens=2000)
    llm = MockLlm([single_text_turn("done", usage)])
    agent = _agent(llm, ctx, default_registry())
    events = await _drain(agent, "expensive query")
    _print_events(events)
    turn_end = [e for e in events if isinstance(e, TurnEnd)][0]
    print(f"  [Cost] ${turn_end.cost_usd:.6f} for 5K input + 2K output tokens")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        await demo_simple_text(tmp)
        await demo_write_then_read(tmp)
        await demo_parallel_tools(tmp)
        await demo_permission_deny(tmp)
        await demo_cost_tracking(tmp)
    print("\n✅ All mock demos completed.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
