"""Subagent 并行执行测试：预设库、派生工具、并发护栏与权限。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_config import PermissionConfig
from mcc_core import ToolContext, Usage
from mcc_core.event import (
    PermissionDecision,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
)
from mcc_core.event import (
    PermissionRequest as PermissionRequestEvent,
)
from mcc_harness import (
    AgentLoopBuilder,
    Allow,
    BudgetAllocation,
    BudgetLedger,
    Deny,
    HookRegistry,
    HookType,
    PermissionAuditEntry,
    PermissionChannel,
    PermissionChannelRequest,
    PermissionChecker,
    SpawnSubagentTool,
    SubagentRunner,
    SubagentSpec,
    SubagentStatus,
    UsageRecord,
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


class ErrorLlm:
    """立即抛出异常的 LLM，用于测试异常终止路径。"""

    async def stream(self, req):
        raise RuntimeError("model crashed")
        yield  # pragma: no cover — 使 stream 成为 async generator


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
            80_000 / 1_000_000 * (0.80 * 0.7 + 4.0 * 0.3)
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
        assert (
            "[subagent=summarizer status=completed tokens=7in/3out reason=end_turn]"
            in out.content
        )
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


# ---------- 统一生命周期出口 (P1-09) ----------


class TestSubagentLifecycle:
    """所有终止路径汇聚到统一出口，返回 SubagentResult with 正确 status。

    验收标准：
    - 成功、超时、取消、模型异常、预算超限均返回对应状态
    - Usage 缺失时为 None（不伪造为零消耗）
    - ON_SUBAGENT_STOP Hook 在每条终止路径触发且只触发一次
    """

    async def test_completed_returns_real_usage(self, registry, ctx):
        """成功路径：status=COMPLETED, usage 为真实值, cost_usd > 0。"""
        llm = MockLlm(
            [single_text_turn("done", Usage(input_tokens=100, output_tokens=50))]
        )
        runner = SubagentRunner(llm, registry)
        result = await runner.run(default_presets()["summarizer"], "task", ctx)
        assert result.status == SubagentStatus.COMPLETED
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.cost_usd > 0
        assert result.error is None
        assert result.stopped_reason == "end_turn"

    async def test_timeout_usage_is_none_not_zero(self, registry, ctx):
        """超时：usage=None（不伪造零消耗），status=TIMEOUT。"""
        spec = SubagentSpec(
            name="slow",
            system="s",
            model="claude-haiku-4-5",
            tools=[],
            timeout_secs=0.05,
        )
        runner = SubagentRunner(SlowLlm(), registry)
        result = await runner.run(spec, "task", ctx)
        assert result.status == SubagentStatus.TIMEOUT
        assert result.usage is None
        assert result.stopped_reason == "timeout"
        assert result.error is None
        assert "timed out" in result.summary

    async def test_timeout_triggers_hook(self, registry, ctx):
        """超时也触发 ON_SUBAGENT_STOP Hook（修复前不触发）。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

        spec = SubagentSpec(
            name="slow",
            system="s",
            model="claude-haiku-4-5",
            tools=[],
            timeout_secs=0.05,
        )
        runner = SubagentRunner(SlowLlm(), registry, hooks=hooks)
        result = await runner.run(spec, "task", ctx)
        assert result.status == SubagentStatus.TIMEOUT
        assert len(hook_calls) == 1
        assert hook_calls[0].is_error is True

    async def test_error_status_and_usage_none(self, registry, ctx):
        """模型异常：status=ERROR, usage=None, error 有值。"""
        runner = SubagentRunner(ErrorLlm(), registry)
        result = await runner.run(default_presets()["summarizer"], "task", ctx)
        assert result.status == SubagentStatus.ERROR
        assert result.usage is None
        assert result.error is not None
        assert "model crashed" in result.error
        assert result.stopped_reason == "error"

    async def test_error_triggers_hook(self, registry, ctx):
        """异常也触发 ON_SUBAGENT_STOP Hook。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

        runner = SubagentRunner(ErrorLlm(), registry, hooks=hooks)
        result = await runner.run(default_presets()["summarizer"], "task", ctx)
        assert result.status == SubagentStatus.ERROR
        assert len(hook_calls) == 1
        assert hook_calls[0].is_error is True

    async def test_budget_exceeded_status(self, registry, ctx):
        """预算超限：status=BUDGET_EXCEEDED, stopped_reason=budget_exceeded。"""
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
        runner = SubagentRunner(llm, registry)
        result = await runner.run(spec, "task", ctx)
        assert result.status == SubagentStatus.BUDGET_EXCEEDED
        assert result.stopped_reason == "budget_exceeded"
        assert result.usage is None
        assert result.error is not None

    async def test_budget_exceeded_triggers_hook(self, registry, ctx):
        """预算超限也触发 ON_SUBAGENT_STOP Hook。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

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
        runner = SubagentRunner(llm, registry, hooks=hooks)
        result = await runner.run(spec, "task", ctx)
        assert result.status == SubagentStatus.BUDGET_EXCEEDED
        assert len(hook_calls) == 1
        assert hook_calls[0].is_error is True

    async def test_cancelled_propagates(self, registry, ctx):
        """取消：CancelledError 传播，不被误判为超时或异常。"""
        runner = SubagentRunner(SlowLlm(), registry)
        spec = default_presets()["summarizer"]

        task_obj = asyncio.create_task(runner.run(spec, "task", ctx))
        await asyncio.sleep(0.01)
        task_obj.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task_obj

    async def test_hook_fires_exactly_once_on_success(self, registry, ctx):
        """Hook 在成功路径只触发一次。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

        runner = SubagentRunner(
            MockLlm([single_text_turn("ok")]), registry, hooks=hooks
        )
        await runner.run(default_presets()["summarizer"], "task", ctx)
        assert len(hook_calls) == 1

    async def test_started_and_finished_at_recorded(self, registry, ctx):
        """started_at 和 finished_at 被记录，且 finished >= started。"""
        llm = MockLlm([single_text_turn("done")])
        runner = SubagentRunner(llm, registry)
        result = await runner.run(default_presets()["summarizer"], "task", ctx)
        assert result.started_at > 0
        assert result.finished_at >= result.started_at

    async def test_hook_context_carries_cost_and_is_error(
        self, registry, ctx
    ):
        """Hook context 携带 cost_usd 和 is_error 标志。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

        llm = MockLlm(
            [single_text_turn("done", Usage(input_tokens=100, output_tokens=50))]
        )
        runner = SubagentRunner(llm, registry, hooks=hooks)
        result = await runner.run(default_presets()["summarizer"], "task", ctx)
        assert len(hook_calls) == 1
        assert hook_calls[0].cost_usd == result.cost_usd
        assert hook_calls[0].cost_usd > 0
        assert hook_calls[0].is_error is False

    @pytest.mark.parametrize(
        "status,expected_is_error",
        [
            (SubagentStatus.COMPLETED, False),
            (SubagentStatus.TIMEOUT, True),
            (SubagentStatus.ERROR, True),
            (SubagentStatus.BUDGET_EXCEEDED, True),
        ],
    )
    async def test_is_error_flag_per_status(
        self, registry, ctx, status, expected_is_error
    ):
        """各终止路径的 Hook context.is_error 正确反映成功/失败。"""
        hook_calls: list = []

        async def on_stop(hook_ctx):
            hook_calls.append(hook_ctx)
            return HookResult()

        hooks = HookRegistry()
        hooks.add_python(HookType.ON_SUBAGENT_STOP, on_stop)

        if status == SubagentStatus.COMPLETED:
            runner = SubagentRunner(
                MockLlm([single_text_turn("ok")]), registry, hooks=hooks
            )
            await runner.run(default_presets()["summarizer"], "task", ctx)
        elif status == SubagentStatus.TIMEOUT:
            spec = SubagentSpec(
                name="slow", system="s", model="claude-haiku-4-5",
                tools=[], timeout_secs=0.05,
            )
            runner = SubagentRunner(SlowLlm(), registry, hooks=hooks)
            await runner.run(spec, "task", ctx)
        elif status == SubagentStatus.ERROR:
            runner = SubagentRunner(ErrorLlm(), registry, hooks=hooks)
            await runner.run(default_presets()["summarizer"], "task", ctx)
        elif status == SubagentStatus.BUDGET_EXCEEDED:
            spec = SubagentSpec(
                name="big", system="s", model="claude-haiku-4-5",
                tools=["read_file"], budget_tokens=1_000,
            )
            turn1 = [
                ToolUseStart(id="r1", name="read_file"),
                ToolUseInputDelta(
                    partial_json=json.dumps({"path": "whatever.txt"})
                ),
                MessageStop(
                    stop_reason="tool_use",
                    usage=Usage(input_tokens=1_000_000, output_tokens=0),
                ),
            ]
            llm = MockLlm([turn1, single_text_turn("x")])
            runner = SubagentRunner(llm, registry, hooks=hooks)
            await runner.run(spec, "task", ctx)

        assert len(hook_calls) == 1
        assert hook_calls[0].is_error is expected_is_error

    async def test_tool_output_unknown_tokens_on_timeout(self, registry, ctx):
        """超时时工具输出 tokens=unknown（usage 未知）。"""
        spec = SubagentSpec(
            name="slowpoke", system="s", model="claude-haiku-4-5",
            tools=[], timeout_secs=0.05,
        )
        tool = _tool(SlowLlm(), registry, {"slowpoke": spec})
        out = await tool.execute({"preset": "slowpoke", "task": "t"}, ctx)
        assert out.is_error
        assert "tokens=unknown" in out.content
        assert "status=timeout" in out.content

    async def test_tool_output_status_on_success(self, registry, ctx):
        """成功时工具输出含 status=completed。"""
        llm = MockLlm(
            [single_text_turn("result", Usage(input_tokens=5, output_tokens=3))]
        )
        tool = _tool(llm, registry)
        out = await tool.execute(
            {"preset": "summarizer", "task": "t"}, ctx
        )
        assert not out.is_error
        assert "status=completed" in out.content


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


# ---------- 权限审批通道 ----------


def _write_turn(path: str = "out.txt", content: str = "x"):
    """构造一个 write_file 工具调用轮次。"""
    return [
        ToolUseStart(id="w1", name="write_file"),
        ToolUseInputDelta(
            partial_json=json.dumps({"path": path, "content": content})
        ),
        MessageStop(stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=5)),
    ]


class TestPermissionChannel:
    """PermissionChannel 单元测试。"""

    async def test_auto_allow_skips_pending(self):
        """委托范围内的请求自动允许，不进入 pending。"""
        ch = PermissionChannel("a1", auto_allow=frozenset({"Write"}))
        ev = PermissionRequestEvent(
            id=42, message="write_file: confirmation required",
            tool_name="write_file",
            tool_input={"path": "x.txt", "content": "y"},
        )
        await ch.handle_event(ev)
        assert ch.pending.empty()
        decision = await asyncio.wait_for(ch.response_queue.get(), timeout=1)
        assert decision.request_id == 42
        assert decision.action == "allow"
        audit = ch.audit_log()
        assert len(audit) == 1
        assert audit[0].decision == "auto_allow"
        assert audit[0].category == "Write"

    async def test_non_auto_allow_goes_to_pending(self):
        """委托范围外的请求进入 pending 等外部响应。"""
        ch = PermissionChannel("a1", auto_allow=frozenset({"Read"}))
        ev = PermissionRequestEvent(
            id=99, message="write_file: confirmation required",
            tool_name="write_file",
            tool_input={"path": "x.txt", "content": "y"},
        )
        await ch.handle_event(ev)
        assert not ch.pending.empty()
        req = await ch.pending.get()
        assert req.request_id == 99
        assert req.category == "Write"
        assert req.tool_name == "write_file"
        assert ch.response_queue.empty()

    async def test_respond_puts_decision(self):
        """外部 respond() 把决策放入 response_queue。"""
        ch = PermissionChannel("a1")
        await ch.respond(PermissionDecision(request_id=7, action="allow"))
        decision = await asyncio.wait_for(ch.response_queue.get(), timeout=1)
        assert decision.request_id == 7
        assert decision.action == "allow"
        audit = ch.audit_log()
        assert len(audit) == 1
        assert audit[0].decision == "allow"

    async def test_audit_records_multiple(self):
        """多次决策都记入审计。"""
        ch = PermissionChannel("a1", auto_allow=frozenset({"Read"}))
        ev1 = PermissionRequestEvent(
            id=1, message="r", tool_name="read_file", tool_input={"path": "a"}
        )
        await ch.handle_event(ev1)
        await ch.respond(PermissionDecision(request_id=2, action="deny"))
        audit = ch.audit_log()
        assert len(audit) == 2
        assert audit[0].decision == "auto_allow"
        assert audit[1].decision == "deny"


class TestSubagentPermissionChannel:
    """子代理权限审批通道集成测试。"""

    async def test_auto_allow_write_executes(
        self, registry, ctx, tmp_path
    ):
        """auto_allow={"Write"} 时子代理的 write_file 自动允许。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_auto_allow=frozenset({"Write"}),
            permission_timeout_secs=5,
        )
        result = await runner.run(spec, "write file", ctx)
        assert result.status == SubagentStatus.COMPLETED
        assert result.summary == "done"
        audit = result.permission_audit
        assert any(e.decision == "auto_allow" and e.category == "Write" for e in audit)

    async def test_no_channel_denies_write(self, registry, ctx, tmp_path):
        """无 auto_allow 且无外部响应时 write_file 超时拒绝。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_auto_allow=frozenset(),
            permission_timeout_secs=0.3,
        )
        result = await runner.run(spec, "write file", ctx)
        assert result.status == SubagentStatus.COMPLETED
        assert not (tmp_path / "out.txt").exists()

    async def test_external_allow_via_channel(self, registry, ctx, tmp_path):
        """外部通过 channel.respond() 允许子代理的 write_file。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_timeout_secs=5,
        )
        channel = PermissionChannel("test-agent", timeout_secs=5)

        async def responder():
            req = await asyncio.wait_for(channel.pending.get(), timeout=5)
            await channel.respond(
                PermissionDecision(request_id=req.request_id, action="allow")
            )

        resp_task = asyncio.create_task(responder())
        result = await runner.run(spec, "write file", ctx, channel=channel)
        await resp_task
        assert result.status == SubagentStatus.COMPLETED
        assert result.summary == "done"

    async def test_external_deny_via_channel(self, registry, ctx, tmp_path):
        """外部通过 channel.respond() 拒绝子代理的 write_file。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_timeout_secs=5,
        )
        channel = PermissionChannel("test-agent", timeout_secs=5)

        async def responder():
            req = await asyncio.wait_for(channel.pending.get(), timeout=5)
            await channel.respond(
                PermissionDecision(request_id=req.request_id, action="deny")
            )

        resp_task = asyncio.create_task(responder())
        result = await runner.run(spec, "write file", ctx, channel=channel)
        await resp_task
        assert result.status == SubagentStatus.COMPLETED
        assert not (tmp_path / "out.txt").exists()

    async def test_permission_timeout_default_deny(
        self, registry, ctx, tmp_path
    ):
        """审批超时后默认拒绝。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_timeout_secs=0.3,
        )
        channel = PermissionChannel("test-agent", timeout_secs=0.3)
        result = await runner.run(spec, "write file", ctx, channel=channel)
        assert result.status == SubagentStatus.COMPLETED
        assert not (tmp_path / "out.txt").exists()

    async def test_concurrent_channels_no_cross_talk(
        self, registry, ctx, tmp_path
    ):
        """两个并发子代理的权限请求互不串线。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm1 = MockLlm([
            _write_turn(str(tmp_path / "a.txt"), "a"),
            single_text_turn("a-done", Usage(input_tokens=5, output_tokens=3)),
        ])
        llm2 = MockLlm([
            _write_turn(str(tmp_path / "b.txt"), "b"),
            single_text_turn("b-done", Usage(input_tokens=5, output_tokens=3)),
        ])
        ch1 = PermissionChannel("agent-1", auto_allow=frozenset({"Write"}))
        ch2 = PermissionChannel("agent-2", auto_allow=frozenset({"Write"}))

        runner1 = SubagentRunner(
            llm1, registry,
            permission=PermissionChecker(PermissionConfig()),
        )
        runner2 = SubagentRunner(
            llm2, registry,
            permission=PermissionChecker(PermissionConfig()),
        )
        r1, r2 = await asyncio.gather(
            runner1.run(spec, "write a", ctx, channel=ch1),
            runner2.run(spec, "write b", ctx, channel=ch2),
        )
        assert r1.status == SubagentStatus.COMPLETED
        assert r2.status == SubagentStatus.COMPLETED
        assert r1.summary == "a-done"
        assert r2.summary == "b-done"
        a1 = ch1.audit_log()
        a2 = ch2.audit_log()
        assert all(e.agent_id == "agent-1" for e in a1)
        assert all(e.agent_id == "agent-2" for e in a2)
        assert len(a1) >= 1
        assert len(a2) >= 1

    async def test_audit_in_result(self, registry, ctx, tmp_path):
        """SubagentResult 携带权限审计日志。"""
        spec = SubagentSpec(
            name="writer", system="s", model="claude-haiku-4-5",
            tools=["write_file"], max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            _write_turn(str(tmp_path / "out.txt"), "hello"),
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_auto_allow=frozenset({"Write"}),
        )
        result = await runner.run(spec, "write file", ctx)
        assert len(result.permission_audit) >= 1
        entry = result.permission_audit[0]
        assert isinstance(entry, PermissionAuditEntry)
        assert entry.decision == "auto_allow"
        assert entry.category == "Write"

    async def test_read_auto_allowed_by_default(
        self, registry, ctx, tmp_path
    ):
        """Read 类别在 PermissionChecker 中默认 Allow，不触发通道。"""
        spec = SubagentSpec(
            name="reader", system="s", model="claude-haiku-4-5",
            tools=["read_file", "list_dir", "grep"],
            max_iterations=2, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        turn1 = [
            ToolUseStart(id="r1", name="read_file"),
            ToolUseInputDelta(
                partial_json=json.dumps({"path": str(tmp_path / "x.txt")})
            ),
            MessageStop(stop_reason="tool_use", usage=Usage(input_tokens=10, output_tokens=5)),
        ]
        llm = MockLlm([
            turn1,
            single_text_turn("read-done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            permission_timeout_secs=5,
        )
        result = await runner.run(spec, "read file", ctx)
        assert result.status == SubagentStatus.COMPLETED
        assert result.permission_audit == []


class TestBudgetLedger:
    """BudgetLedger 单元测试。"""

    async def test_allocate_success(self):
        ledger = BudgetLedger(total_tokens=100_000)
        alloc = await ledger.allocate("agent-1", 40_000)
        assert alloc is not None
        assert alloc.agent_id == "agent-1"
        assert alloc.allocated_tokens == 40_000
        assert alloc.consumed_tokens == 0

    async def test_allocate_insufficient(self):
        ledger = BudgetLedger(total_tokens=50_000)
        a1 = await ledger.allocate("agent-1", 30_000)
        assert a1 is not None
        a2 = await ledger.allocate("agent-2", 30_000)
        assert a2 is None

    async def test_allocate_exact_fit(self):
        ledger = BudgetLedger(total_tokens=60_000)
        a1 = await ledger.allocate("agent-1", 30_000)
        a2 = await ledger.allocate("agent-2", 30_000)
        assert a1 is not None
        assert a2 is not None

    async def test_remaining_tokens(self):
        ledger = BudgetLedger(total_tokens=100_000)
        await ledger.allocate("agent-1", 30_000)
        await ledger.allocate("agent-2", 20_000)
        remaining = await ledger.remaining_tokens()
        assert remaining == 50_000

    async def test_release_reclaims(self):
        ledger = BudgetLedger(total_tokens=100_000)
        await ledger.allocate("agent-1", 40_000)
        await ledger.release("agent-1")
        remaining = await ledger.remaining_tokens()
        assert remaining == 100_000

    async def test_report_consumption(self):
        ledger = BudgetLedger(total_tokens=100_000)
        await ledger.allocate("agent-1", 50_000)
        usage = Usage(input_tokens=10_000, output_tokens=5_000)
        await ledger.report("agent-1", "claude-haiku-4-5", usage, 0.02, 1.5)
        consumed = await ledger.total_consumed_tokens()
        assert consumed == 15_000
        cost = await ledger.total_consumed_cost_usd()
        assert cost == pytest.approx(0.02)

    async def test_is_exceeded_false(self):
        ledger = BudgetLedger(total_tokens=100_000)
        await ledger.allocate("agent-1", 50_000)
        await ledger.report(
            "agent-1", "m", Usage(input_tokens=10, output_tokens=5), 0.0, 1.0
        )
        assert not await ledger.is_exceeded()

    async def test_is_exceeded_true(self):
        ledger = BudgetLedger(total_tokens=100)
        await ledger.allocate("agent-1", 100)
        await ledger.report(
            "agent-1", "m", Usage(input_tokens=80, output_tokens=50), 0.0, 1.0
        )
        assert await ledger.is_exceeded()

    async def test_summary_by_agent(self):
        ledger = BudgetLedger(total_tokens=1_000_000)
        await ledger.allocate("agent-1", 100_000)
        await ledger.report(
            "agent-1", "claude-haiku-4-5",
            Usage(input_tokens=100, output_tokens=50), 0.001, 1.0,
        )
        await ledger.report(
            "agent-1", "claude-haiku-4-5",
            Usage(input_tokens=200, output_tokens=100), 0.002, 2.0,
        )
        summary = await ledger.summary_by_agent()
        assert "agent-1" in summary
        rec = summary["agent-1"]
        assert rec.usage.input_tokens == 300
        assert rec.usage.output_tokens == 150
        assert rec.cost_usd == pytest.approx(0.003)

    async def test_summary_by_model(self):
        ledger = BudgetLedger(total_tokens=1_000_000)
        await ledger.allocate("agent-1", 100_000)
        await ledger.allocate("agent-2", 100_000)
        await ledger.report(
            "agent-1", "claude-haiku-4-5",
            Usage(input_tokens=100, output_tokens=50), 0.001, 1.0,
        )
        await ledger.report(
            "agent-2", "claude-haiku-4-5",
            Usage(input_tokens=200, output_tokens=100), 0.002, 2.0,
        )
        summary = await ledger.summary_by_model()
        assert "claude-haiku-4-5" in summary
        rec = summary["claude-haiku-4-5"]
        assert rec.usage.input_tokens == 300
        assert rec.usage.output_tokens == 150

    async def test_records_stored(self):
        ledger = BudgetLedger(total_tokens=1_000_000)
        await ledger.allocate("agent-1", 100_000)
        await ledger.report(
            "agent-1", "m", Usage(input_tokens=10, output_tokens=5), 0.01, 0.5,
        )
        records = ledger.records()
        assert len(records) == 1
        assert isinstance(records[0], UsageRecord)
        assert records[0].agent_id == "agent-1"


class TestBudgetAllocation:
    """BudgetAllocation 单元测试。"""

    def test_remaining_tokens(self):
        alloc = BudgetAllocation(
            agent_id="a", allocated_tokens=1000, consumed_tokens=300
        )
        assert alloc.remaining_tokens == 700

    def test_remaining_tokens_floor_zero(self):
        alloc = BudgetAllocation(
            agent_id="a", allocated_tokens=1000, consumed_tokens=1500
        )
        assert alloc.remaining_tokens == 0

    def test_is_exceeded_false(self):
        alloc = BudgetAllocation(
            agent_id="a", allocated_tokens=1000, consumed_tokens=500
        )
        assert not alloc.is_exceeded

    def test_is_exceeded_true(self):
        alloc = BudgetAllocation(
            agent_id="a", allocated_tokens=1000, consumed_tokens=1001
        )
        assert alloc.is_exceeded


class TestTokenBudgetToUsd:
    """token_budget_to_usd 加权计算测试。"""

    def test_default_ratio_07(self):
        usd = token_budget_to_usd(1_000_000, "claude-haiku-4-5")
        in_m, out_m = 0.80, 4.0
        expected = in_m * 0.7 + out_m * 0.3
        assert usd == pytest.approx(expected)

    def test_custom_ratio(self):
        usd = token_budget_to_usd(
            1_000_000, "claude-haiku-4-5", input_ratio=0.5
        )
        in_m, out_m = 0.80, 4.0
        expected = in_m * 0.5 + out_m * 0.5
        assert usd == pytest.approx(expected)

    def test_ratio_affects_result(self):
        usd_07 = token_budget_to_usd(1_000_000, "claude-haiku-4-5")
        usd_03 = token_budget_to_usd(
            1_000_000, "claude-haiku-4-5", input_ratio=0.3
        )
        assert usd_03 > usd_07


class TestSubagentBudgetLedger:
    """SubagentRunner + BudgetLedger 集成测试。"""

    async def test_ledger_tracks_consumption(
        self, registry, ctx, tmp_path
    ):
        """子代理完成后，账本记录实际消耗。"""
        spec = SubagentSpec(
            name="worker", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            single_text_turn("done", Usage(input_tokens=100, output_tokens=50)),
        ])
        ledger = BudgetLedger(total_tokens=100_000)
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            budget_ledger=ledger,
        )
        result = await runner.run(spec, "say done", ctx)
        assert result.status == SubagentStatus.COMPLETED
        consumed = await ledger.total_consumed_tokens()
        assert consumed == 150
        records = ledger.records()
        assert len(records) == 1
        assert records[0].model == "claude-haiku-4-5"

    async def test_budget_exhausted_returns_budget_exceeded(
        self, registry, ctx
    ):
        """预算耗尽时返回 BUDGET_EXCEEDED。"""
        spec = SubagentSpec(
            name="worker", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=50_000, timeout_secs=10,
        )
        llm = MockLlm([
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        ledger = BudgetLedger(total_tokens=60_000)
        await ledger.allocate("other-agent", 50_000)
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            budget_ledger=ledger,
        )
        result = await runner.run(spec, "say done", ctx)
        assert result.status == SubagentStatus.BUDGET_EXCEEDED
        assert "budget exhausted" in result.summary

    async def test_concurrent_subagents_share_budget(
        self, registry, ctx, tmp_path
    ):
        """并发子代理共享同一预算账本，总分配不超过上限。"""
        spec1 = SubagentSpec(
            name="worker-a", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=30_000, timeout_secs=10,
        )
        spec2 = SubagentSpec(
            name="worker-b", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=30_000, timeout_secs=10,
        )
        llm1 = MockLlm([
            single_text_turn("a", Usage(input_tokens=100, output_tokens=50)),
        ])
        llm2 = MockLlm([
            single_text_turn("b", Usage(input_tokens=200, output_tokens=100)),
        ])
        ledger = BudgetLedger(total_tokens=50_000)
        runner1 = SubagentRunner(
            llm1, registry,
            permission=PermissionChecker(PermissionConfig()),
            budget_ledger=ledger,
        )
        runner2 = SubagentRunner(
            llm2, registry,
            permission=PermissionChecker(PermissionConfig()),
            budget_ledger=ledger,
        )
        r1, r2 = await asyncio.gather(
            runner1.run(spec1, "a", ctx),
            runner2.run(spec2, "b", ctx),
        )
        assert r1.status == SubagentStatus.COMPLETED
        assert r2.status == SubagentStatus.BUDGET_EXCEEDED
        consumed = await ledger.total_consumed_tokens()
        assert consumed == 150

    async def test_release_reclaims_after_completion(
        self, registry, ctx
    ):
        """子代理完成后释放分配，额度可复用。"""
        spec = SubagentSpec(
            name="worker", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=40_000, timeout_secs=10,
        )
        llm = MockLlm([
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        ledger = BudgetLedger(total_tokens=50_000)
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
            budget_ledger=ledger,
        )
        await runner.run(spec, "first", ctx)
        remaining = await ledger.remaining_tokens()
        assert remaining == 50_000
        result2 = await runner.run(spec, "second", ctx)
        assert result2.status == SubagentStatus.COMPLETED

    async def test_no_ledger_works_as_before(self, registry, ctx):
        """无账本时行为与之前一致。"""
        spec = SubagentSpec(
            name="worker", system="s", model="claude-haiku-4-5",
            tools=[], max_iterations=1, max_tokens=256,
            budget_tokens=10_000, timeout_secs=10,
        )
        llm = MockLlm([
            single_text_turn("done", Usage(input_tokens=5, output_tokens=3)),
        ])
        runner = SubagentRunner(
            llm, registry,
            permission=PermissionChecker(PermissionConfig()),
        )
        result = await runner.run(spec, "say done", ctx)
        assert result.status == SubagentStatus.COMPLETED
        assert result.summary == "done"
