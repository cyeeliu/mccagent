"""Subagent 并行执行测试：预设库、派生工具、并发护栏与权限。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_config import PermissionConfig
from mcc_core import ToolContext, Usage
from mcc_core.event import TextDelta, ToolCallEnd, ToolCallStart
from mcc_harness import (
    AgentLoopBuilder,
    Allow,
    Deny,
    HookRegistry,
    HookType,
    PermissionChecker,
    SpawnSubagentTool,
    SubagentRunner,
    SubagentSpec,
    default_presets,
    permission_request_for,
    token_budget_to_usd,
)
from mcc_harness.hooks import HookResult
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="e2e", depth=0)


@pytest.fixture
def registry():
    return default_registry()


def _tool(llm, registry, presets=None, max_depth=3, **runner_kwargs):
    runner = SubagentRunner(llm, registry, **runner_kwargs)
    return SpawnSubagentTool(runner, presets or default_presets(), max_depth)


class RecordingLlm:
    """记录调用区间、可选延迟的 LLM，用于并发顺序断言。"""

    def __init__(self, events: list, delay: float = 0.02) -> None:
        self.events = events
        self.delay = delay

    async def stream(self, req):
        self.events.append("start")
        await asyncio.sleep(self.delay)
        yield TextDelta(text="ok")
        yield MessageStop(
            stop_reason="end_turn", usage=Usage(input_tokens=1, output_tokens=1)
        )
        self.events.append("end")


class SlowLlm:
    """永远超时的 LLM。"""

    async def stream(self, req):
        yield TextDelta(text="slow")
        await asyncio.sleep(10)
        yield MessageStop(stop_reason="end_turn", usage=Usage())


# ---------- 预设库 ----------


class TestDefaultPresets:
    def test_contains_three_roles(self):
        presets = default_presets()
        assert set(presets) == {"code_explorer", "test_runner", "summarizer"}

    def test_models_injected(self):
        presets = default_presets("sub-m", "sum-m")
        assert presets["code_explorer"].model == "sub-m"
        assert presets["test_runner"].model == "sub-m"
        assert presets["summarizer"].model == "sum-m"

    def test_summarizer_falls_back_to_subagent_model(self):
        presets = default_presets("sub-m")
        assert presets["summarizer"].model == "sub-m"

    def test_code_explorer_is_read_only(self):
        presets = default_presets()
        assert set(presets["code_explorer"].tools) == {
            "read_file",
            "list_dir",
            "grep",
        }

    def test_summarizer_has_no_tools(self):
        presets = default_presets()
        assert presets["summarizer"].tools == []
        assert presets["summarizer"].max_iterations == 1

    def test_guardrails_present(self):
        for spec in default_presets().values():
            assert spec.timeout_secs > 0
            assert spec.budget_tokens > 0
            assert spec.max_iterations >= 1


# ---------- 预算换算 ----------


class TestTokenBudget:
    def test_haiku_budget(self):
        assert token_budget_to_usd(80_000, "claude-haiku-4-5") == pytest.approx(
            80_000 / 1_000_000 * (0.80 + 4.0) / 2
        )

    def test_budget_scales_linearly(self):
        small = token_budget_to_usd(1_000, "claude-haiku-4-5")
        large = token_budget_to_usd(10_000, "claude-haiku-4-5")
        assert large == pytest.approx(small * 10)


# ---------- SpawnSubagentTool 契约 ----------


class TestSpawnSubagentToolContract:
    def test_schema(self, registry):
        tool = _tool(MockLlm([]), registry)
        schema = tool.input_schema()
        assert schema["required"] == ["preset", "task"]
        assert set(schema["properties"]["preset"]["enum"]) == {
            "code_explorer",
            "test_runner",
            "summarizer",
        }
        assert tool.name == "spawn_subagent"

    async def test_depth_limit(self, registry, ctx):
        tool = _tool(MockLlm([]), registry, max_depth=3)
        deep = ToolContext(cwd=ctx.cwd, session_id="s", depth=3)
        out = await tool.execute(
            {"preset": "summarizer", "task": "t"}, deep
        )
        assert out.is_error
        assert "depth limit" in out.content

    async def test_unknown_preset(self, registry, ctx):
        tool = _tool(MockLlm([]), registry)
        out = await tool.execute({"preset": "nope", "task": "t"}, ctx)
        assert out.is_error
        assert "unknown preset" in out.content

    @pytest.mark.parametrize(
        "payload",
        [
            {"preset": "summarizer"},
            {"task": "t"},
            {},
        ],
    )
    async def test_missing_fields(self, registry, ctx, payload):
        tool = _tool(MockLlm([]), registry)
        out = await tool.execute(payload, ctx)
        assert out.is_error
        assert "required" in out.content

    async def test_ok_result_format(self, registry, ctx):
        llm = MockLlm(
            [single_text_turn("found entrypoint", Usage(input_tokens=7, output_tokens=3))]
        )
        tool = _tool(llm, registry)
        out = await tool.execute(
            {"preset": "summarizer", "task": "find entrypoint"}, ctx
        )
        assert not out.is_error
        assert "[subagent=summarizer tokens=7in/3out reason=end_turn]" in out.content
        assert "found entrypoint" in out.content


# ---------- SubagentRunner 行为 ----------


class TestSubagentRunner:
    async def test_summary_usage_and_reason(self, registry, ctx):
        llm = MockLlm(
            [single_text_turn("结论", Usage(input_tokens=100, output_tokens=50))]
        )
        runner = SubagentRunner(llm, registry)
        result = await runner.run(default_presets()["summarizer"], "任务", ctx)
        assert result.name == "summarizer"
        assert result.summary == "结论"
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.stopped_reason == "end_turn"

    async def test_session_id_and_depth_propagated(self, registry, ctx):
        seen: list = []

        async def on_stop(hook_ctx):
            seen.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)
        runner = SubagentRunner(MockLlm([single_text_turn("ok")]), registry, hooks=hooks)

        await runner.run(default_presets()["code_explorer"], "任务", ctx)

        assert len(seen) == 1
        assert seen[0].session_id == "e2e::code_explorer"
        assert seen[0].depth == 1

    async def test_hook_error_not_fatal(self, registry, ctx):
        async def boom(hook_ctx):
            raise RuntimeError("hook boom")

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, boom)
        runner = SubagentRunner(MockLlm([single_text_turn("ok")]), registry, hooks=hooks)

        result = await runner.run(default_presets()["summarizer"], "任务", ctx)
        assert result.stopped_reason == "end_turn"

    async def test_child_registry_excludes_spawn(self, registry, ctx, tmp_path):
        """子代理内再调 spawn_subagent 应为 unknown tool（防递归）。"""
        (tmp_path / "x.txt").write_text("hi", encoding="utf-8")
        turn1 = [
            ToolUseStart(id="s1", name="spawn_subagent"),
            ToolUseInputDelta(
                partial_json=json.dumps({"preset": "summarizer", "task": "x"})
            ),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]
        llm = MockLlm([turn1, single_text_turn("child done")])
        runner = SubagentRunner(llm, registry)
        result = await runner.run(default_presets()["code_explorer"], "任务", ctx)
        assert result.summary == "child done"

    async def test_child_can_use_allowed_tools(self, registry, ctx, tmp_path):
        """子代理的 subset 注册表可用（read_file 正常执行）。"""
        (tmp_path / "note.txt").write_text("payload", encoding="utf-8")
        turn1 = [
            ToolUseStart(id="r1", name="read_file"),
            ToolUseInputDelta(
                partial_json=json.dumps({"path": "note.txt"})
            ),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]
        llm = MockLlm([turn1, single_text_turn("read: payload")])
        runner = SubagentRunner(llm, registry)
        result = await runner.run(default_presets()["code_explorer"], "任务", ctx)
        assert "read: payload" in result.summary

    async def test_timeout(self, registry, ctx):
        spec = SubagentSpec(
            name="slowpoke",
            system="s",
            model="claude-haiku-4-5",
            tools=[],
            timeout_secs=0.05,
        )
        tool = _tool(SlowLlm(), registry, {"slowpoke": spec})
        out = await tool.execute({"preset": "slowpoke", "task": "t"}, ctx)
        assert out.is_error
        assert "reason=timeout" in out.content
        assert "timed out" in out.content

    async def test_budget_exceeded(self, registry, ctx):
        spec = SubagentSpec(
            name="big_spender",
            system="s",
            model="claude-haiku-4-5",
            tools=["read_file"],
            budget_tokens=1_000,
        )
        turn1 = [
            ToolUseStart(id="r1", name="read_file"),
            ToolUseInputDelta(partial_json=json.dumps({"path": "whatever.txt"})),
            MessageStop(
                stop_reason="tool_use",
                usage=Usage(input_tokens=1_000_000, output_tokens=0),
            ),
        ]
        llm = MockLlm([turn1, single_text_turn("never reached")])
        tool = _tool(llm, registry, {"big_spender": spec})
        out = await tool.execute({"preset": "big_spender", "task": "t"}, ctx)
        assert out.is_error
        assert "subagent failed" in out.content

    async def test_hooks_passed_to_child(self, registry, ctx, tmp_path):
        """子代理共享主钩子注册表：POST_TOOL 可观察到子代理内工具执行。"""
        (tmp_path / "note.txt").write_text("child-payload", encoding="utf-8")
        outputs: list = []

        async def post_tool(hook_ctx):
            outputs.append(
                (hook_ctx.tool_name, hook_ctx.tool_output, hook_ctx.is_error, hook_ctx.depth)
            )
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.POST_TOOL, post_tool)

        turn1 = [
            ToolUseStart(id="r1", name="read_file"),
            ToolUseInputDelta(partial_json=json.dumps({"path": "note.txt"})),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]
        llm = MockLlm([turn1, single_text_turn("ok")])
        runner = SubagentRunner(llm, registry, hooks=hooks)

        result = await runner.run(default_presets()["code_explorer"], "任务", ctx)

        assert result.stopped_reason == "end_turn"
        name, output, is_error, depth = outputs[0]
        assert name == "read_file"
        assert "child-payload" in output
        assert not is_error
        assert depth == 1

    async def test_permission_deny_in_child(self, registry, ctx, tmp_path):
        """子代理继承主权限检查器：deny 规则使工具在执行前被拒。"""
        (tmp_path / "secret.txt").write_text("s3cret", encoding="utf-8")
        outputs: list = []

        async def post_tool(hook_ctx):
            outputs.append((hook_ctx.tool_name, hook_ctx.tool_output, hook_ctx.is_error))
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.POST_TOOL, post_tool)

        checker = PermissionChecker(PermissionConfig(deny=["Read(secret.txt)"]))
        turn1 = [
            ToolUseStart(id="r1", name="read_file"),
            ToolUseInputDelta(partial_json=json.dumps({"path": "secret.txt"})),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]
        llm = MockLlm([turn1, single_text_turn("denied")])
        runner = SubagentRunner(llm, registry, hooks=hooks, permission=checker)

        result = await runner.run(default_presets()["code_explorer"], "任务", ctx)

        assert result.stopped_reason == "end_turn"
        assert result.summary == "denied"
        assert outputs == []


# ---------- 并发护栏 ----------


class TestConcurrency:
    async def test_concurrency_one_serializes(self, registry, ctx):
        events: list = []
        runner = SubagentRunner(RecordingLlm(events), registry, concurrency=1)
        spec = default_presets()["summarizer"]
        await asyncio.gather(
            runner.run(spec, "a", ctx), runner.run(spec, "b", ctx)
        )
        assert events == ["start", "end", "start", "end"]

    async def test_concurrency_two_overlaps(self, registry, ctx):
        events: list = []
        runner = SubagentRunner(RecordingLlm(events), registry, concurrency=2)
        spec = default_presets()["summarizer"]
        await asyncio.gather(
            runner.run(spec, "a", ctx), runner.run(spec, "b", ctx)
        )
        assert events[:2] == ["start", "start"]


# ---------- 权限映射 ----------


class TestSpawnPermission:
    def test_request_mapping(self):
        req = permission_request_for(
            "spawn_subagent", {"preset": "code_explorer", "task": "t"}
        )
        assert req.category == "Subagent"
        assert req.action.cmd == "spawn_subagent code_explorer"

    def test_default_allows_spawn(self):
        checker = PermissionChecker(PermissionConfig())
        req = permission_request_for(
            "spawn_subagent", {"preset": "code_explorer", "task": "t"}
        )
        assert isinstance(checker.check(req), Allow)

    def test_deny_all_spawns(self):
        checker = PermissionChecker(PermissionConfig(deny=["Bash(spawn_subagent:*)"]))
        req = permission_request_for(
            "spawn_subagent", {"preset": "code_explorer", "task": "t"}
        )
        assert isinstance(checker.check(req), Deny)

    def test_deny_specific_preset_only(self):
        checker = PermissionChecker(
            PermissionConfig(deny=["Bash(spawn_subagent test_runner:*)"])
        )
        blocked = permission_request_for(
            "spawn_subagent", {"preset": "test_runner", "task": "t"}
        )
        allowed = permission_request_for(
            "spawn_subagent", {"preset": "code_explorer", "task": "t"}
        )
        assert isinstance(checker.check(blocked), Deny)
        assert isinstance(checker.check(allowed), Allow)


# ---------- Fan-out 端到端 ----------


class TestFanOutE2E:
    async def test_four_way_fanout(self, registry, ctx):
        """主代理单轮发出 4 个 spawn_subagent，并行汇总。"""
        tasks = ["entrypoints", "core modules", "io layer", "tests"]
        turn1 = []
        for i, task in enumerate(tasks):
            turn1.append(ToolUseStart(id=f"sp{i}", name="spawn_subagent"))
            turn1.append(
                ToolUseInputDelta(
                    partial_json=json.dumps(
                        {"preset": "code_explorer", "task": task}
                    )
                )
            )
        turn1.append(MessageStop(stop_reason="tool_use", usage=Usage()))
        child_turns = [
            single_text_turn(f"summary: {t}", Usage(input_tokens=10, output_tokens=5))
            for t in tasks
        ]
        final = single_text_turn("architecture overview")
        llm = MockLlm([turn1, *child_turns, final])

        registry.register(_tool(llm, registry))
        agent = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5").build()

        from mcc_cli.events import AgentEventSource

        events = []
        async for ev in AgentEventSource(agent).run("analyze repo"):
            events.append(ev)

        ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(ends) == 4
        assert all(not e.is_error for e in ends)

        summaries = {e.output.splitlines()[-1] for e in ends}
        assert summaries == {f"summary: {t}" for t in tasks}

        starts = [e for e in events if isinstance(e, ToolCallStart)]
        assert all(s.name == "spawn_subagent" for s in starts)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "architecture overview"
