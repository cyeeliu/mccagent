"""统一 AgentRuntime 抽象测试：接口契约 + Ephemeral/Persistent 生命周期 + 旧 API 兼容。

P2-01 验收标准：
- 两种 Runtime 均通过同一接口启动、停止和查询状态。
- 旧 API 可通过兼容适配器继续工作。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import ToolContext, Usage
from mcc_harness import (
    LEAD_NAME,
    AgentRuntime,
    EphemeralRuntime,
    MessageBus,
    PersistentRuntime,
    PlanStore,
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    SubagentRunner,
    SubagentSpec,
    SubagentStatus,
    TeammateRuntime,
    TeammateSpec,
    default_presets,
)
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry, default_registry

# -- helpers --


def _make_ephemeral(
    tmp_path: Path,
    llm=None,
    spec_name: str = "explorer",
) -> EphemeralRuntime:
    registry = default_registry()
    runner = SubagentRunner(llm or MockLlm([single_text_turn("ok")]), registry)
    presets = default_presets()
    spec = presets.get(spec_name, SubagentSpec(
        name=spec_name, system="test", model="claude-haiku-4-5",
    ))
    ctx = ToolContext(cwd=tmp_path, session_id="rt-test", depth=0)
    return EphemeralRuntime(runner, spec, ctx)


def _make_persistent(
    tmp_path: Path,
    llm=None,
    name: str = "worker",
) -> PersistentRuntime:
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    registry = ToolRegistry()
    spec = TeammateSpec(
        name=name, role="coder", system="test", model="claude-haiku-4-5",
    )
    rt = TeammateRuntime(
        spec=spec,
        llm=llm or MockLlm([single_text_turn("done")]),
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        base_dir=tmp_path,
    )
    return PersistentRuntime(rt, bus)


# -- 接口契约测试 --


class TestRuntimeContract:
    def test_runtime_mode_values(self):
        assert RuntimeMode.EPHEMERAL == "ephemeral"
        assert RuntimeMode.PERSISTENT == "persistent"

    def test_runtime_status_is_str_enum(self):
        assert RuntimeStatus.CREATED == "created"
        assert RuntimeStatus.RUNNING == "running"
        assert RuntimeStatus.SHUTDOWN == "shutdown"

    @pytest.mark.parametrize("status,expected", [
        (RuntimeStatus.COMPLETED, True),
        (RuntimeStatus.FAILED, True),
        (RuntimeStatus.CANCELLED, True),
        (RuntimeStatus.TIMEOUT, True),
        (RuntimeStatus.BUDGET_EXCEEDED, True),
        (RuntimeStatus.SHUTDOWN, True),
        (RuntimeStatus.CREATED, False),
        (RuntimeStatus.RUNNING, False),
        (RuntimeStatus.IDLE, False),
        (RuntimeStatus.PAUSED, False),
    ])
    def test_result_is_terminal(self, status, expected):
        r = RuntimeResult(runtime_id="x", status=status, summary="")
        assert r.is_terminal is expected

    def test_ephemeral_satisfies_protocol(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        assert isinstance(rt, AgentRuntime)

    def test_persistent_satisfies_protocol(self, tmp_path):
        rt = _make_persistent(tmp_path)
        assert isinstance(rt, AgentRuntime)

    def test_config_ephemeral_mode(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        cfg = rt.config
        assert cfg.mode is RuntimeMode.EPHEMERAL
        assert isinstance(cfg, RuntimeConfig)

    def test_config_persistent_mode(self, tmp_path):
        rt = _make_persistent(tmp_path)
        cfg = rt.config
        assert cfg.mode is RuntimeMode.PERSISTENT
        assert isinstance(cfg, RuntimeConfig)


# -- EphemeralRuntime 生命周期测试 --


class TestEphemeralRuntime:
    async def test_start_and_wait_done(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        assert rt.status is RuntimeStatus.CREATED
        assert not rt.alive

        await rt.start("explore the code")
        assert rt.alive
        assert rt.status is RuntimeStatus.RUNNING

        result = await rt.wait_done()
        assert result.is_terminal
        assert result.status is RuntimeStatus.COMPLETED
        assert result.runtime_id == rt.runtime_id
        assert not rt.alive

    async def test_cancel(self, tmp_path):
        llm = MockLlm([single_text_turn("first"), single_text_turn("second")])
        rt = _make_ephemeral(tmp_path, llm=llm)
        await rt.start("long task")
        rt.cancel()
        result = await rt.wait_done()
        assert result.status in (RuntimeStatus.CANCELLED, RuntimeStatus.COMPLETED)

    async def test_stop_cancels_running(self, tmp_path):
        llm = MockLlm([single_text_turn("first"), single_text_turn("second")])
        rt = _make_ephemeral(tmp_path, llm=llm)
        await rt.start("task")
        await rt.stop(timeout=5.0)
        assert not rt.alive
        assert rt.status in (
            RuntimeStatus.CANCELLED,
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
        )

    async def test_double_start_raises(self, tmp_path):
        llm = MockLlm([single_text_turn("first"), single_text_turn("second")])
        rt = _make_ephemeral(tmp_path, llm=llm)
        await rt.start("task1")
        with pytest.raises(RuntimeError, match="already running"):
            await rt.start("task2")
        await rt.wait_done()

    async def test_pause_not_supported(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        with pytest.raises(RuntimeError, match="pause"):
            rt.pause()

    async def test_resume_not_supported(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        with pytest.raises(RuntimeError, match="resume"):
            rt.resume()

    async def test_inject_instruction_not_supported(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        with pytest.raises(RuntimeError, match="instruction"):
            rt.inject_instruction("do something")

    async def test_wait_done_without_start(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        result = await rt.wait_done()
        assert result.status is RuntimeStatus.CREATED
        assert "not started" in result.summary

    async def test_wait_done_with_timeout(self, tmp_path):
        llm = MockLlm([single_text_turn("first"), single_text_turn("second")])
        rt = _make_ephemeral(tmp_path, llm=llm)
        await rt.start("task")
        result = await rt.wait_done(timeout=10.0)
        assert result.is_terminal

    async def test_runtime_id_format(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        assert "rt-test" in rt.runtime_id
        assert "::" in rt.runtime_id

    async def test_config_reflects_spec(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        cfg = rt.config
        assert cfg.name == "explorer"
        assert cfg.timeout_secs > 0
        assert cfg.budget_tokens > 0

    async def test_status_transitions(self, tmp_path):
        rt = _make_ephemeral(tmp_path)
        assert rt.status is RuntimeStatus.CREATED
        await rt.start("task")
        assert rt.status is RuntimeStatus.RUNNING
        await rt.wait_done()
        assert rt.status is RuntimeStatus.COMPLETED


# -- PersistentRuntime 生命周期测试 --


class TestPersistentRuntime:
    async def test_start_and_status(self, tmp_path):
        rt = _make_persistent(tmp_path)
        assert rt.status is RuntimeStatus.CREATED
        assert not rt.alive

        await rt.start("do work")
        assert rt.alive
        await asyncio.sleep(0.3)
        assert rt.status in (
            RuntimeStatus.RUNNING,
            RuntimeStatus.IDLE,
            RuntimeStatus.STARTING,
        )
        await rt.stop()
        assert not rt.alive

    async def test_stop_returns_shutdown(self, tmp_path):
        rt = _make_persistent(tmp_path)
        await rt.start("do work")
        await asyncio.sleep(0.1)
        await rt.stop()
        assert rt.status is RuntimeStatus.SHUTDOWN

    async def test_wait_done_after_stop(self, tmp_path):
        rt = _make_persistent(tmp_path)
        await rt.start("do work")
        await asyncio.sleep(0.1)
        await rt.stop()
        result = await rt.wait_done()
        assert result.is_terminal
        assert result.status is RuntimeStatus.SHUTDOWN
        assert result.runtime_id == rt.runtime_id

    async def test_cancel_sends_signal(self, tmp_path):
        rt = _make_persistent(tmp_path)
        await rt.start("do work")
        await asyncio.sleep(0.1)
        rt.cancel()
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_pause_and_resume(self, tmp_path):
        llm = MockLlm([
            single_text_turn("first"),
            single_text_turn("second"),
        ])
        rt = _make_persistent(tmp_path, llm=llm, name="pausable")
        await rt.start("first task")
        await asyncio.sleep(0.2)
        rt.pause()
        await asyncio.sleep(0.1)
        rt.resume()
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_inject_instruction(self, tmp_path):
        llm = MockLlm([
            single_text_turn("first"),
            single_text_turn("second"),
        ])
        rt = _make_persistent(tmp_path, llm=llm, name="injectable")
        await rt.start("initial task")
        await asyncio.sleep(0.1)
        rt.inject_instruction("extra guidance")
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_double_start_raises(self, tmp_path):
        llm = MockLlm([
            single_text_turn("first"),
            single_text_turn("second"),
        ])
        rt = _make_persistent(tmp_path, llm=llm)
        await rt.start("task1")
        with pytest.raises(RuntimeError, match="already running"):
            await rt.start("task2")
        await rt.stop()

    async def test_wait_done_without_start(self, tmp_path):
        rt = _make_persistent(tmp_path)
        result = await rt.wait_done()
        assert result.status is RuntimeStatus.SHUTDOWN
        assert "not started" in result.summary

    async def test_runtime_id_is_teammate_name(self, tmp_path):
        rt = _make_persistent(tmp_path, name="my-worker")
        assert rt.runtime_id == "my-worker"

    async def test_config_reflects_spec(self, tmp_path):
        rt = _make_persistent(tmp_path, name="cfg-test")
        cfg = rt.config
        assert cfg.name == "cfg-test"
        assert cfg.mode is RuntimeMode.PERSISTENT
        assert cfg.require_plan is False

    async def test_cancel_when_not_alive_is_noop(self, tmp_path):
        rt = _make_persistent(tmp_path)
        rt.cancel()
        rt.pause()
        rt.resume()
        rt.inject_instruction("test")

    async def test_work_result_via_bus(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="bus-test", role="coder", system="test",
            model="claude-haiku-4-5",
        )
        teammate = TeammateRuntime(
            spec=spec,
            llm=MockLlm([single_text_turn("work complete")]),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        rt = PersistentRuntime(teammate, bus)
        await rt.start("do the work")
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        assert any(m.kind == "work_done" for m in msgs)
        await rt.stop()


# -- 旧 API 兼容测试 --


class TestOldApiCompatibility:
    async def test_subagent_runner_direct(self, tmp_path):
        registry = default_registry()
        llm = MockLlm([single_text_turn("result")])
        runner = SubagentRunner(llm, registry)
        spec = default_presets()["code_explorer"]
        ctx = ToolContext(cwd=tmp_path, session_id="compat", depth=0)
        result = await runner.run(spec, "explore", ctx)
        assert result.status is SubagentStatus.COMPLETED
        assert result.summary == "result"

    async def test_teammate_runtime_direct(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="compat", role="coder", system="test",
            model="claude-haiku-4-5",
        )
        rt = TeammateRuntime(
            spec=spec,
            llm=MockLlm([single_text_turn("done")]),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        await rt.start("do work")
        assert rt.alive
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        assert any(m.kind == "work_done" for m in msgs)
        await rt.shutdown()
        assert not rt.alive

    async def test_ephemeral_wraps_subagent_consistent(self, tmp_path):
        registry = default_registry()
        llm = MockLlm([single_text_turn("via runtime")])
        runner = SubagentRunner(llm, registry)
        spec = default_presets()["code_explorer"]
        ctx = ToolContext(cwd=tmp_path, session_id="cons", depth=0)

        direct = await runner.run(spec, "explore", ctx)

        runner2 = SubagentRunner(
            MockLlm([single_text_turn("via runtime")]), registry,
        )
        rt = EphemeralRuntime(runner2, spec, ctx)
        await rt.start("explore")
        wrapped = await rt.wait_done()

        assert direct.status is SubagentStatus.COMPLETED
        assert wrapped.status is RuntimeStatus.COMPLETED
        assert direct.summary == wrapped.summary

    async def test_persistent_wraps_teammate_consistent(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="cons", role="coder", system="test",
            model="claude-haiku-4-5",
        )
        teammate = TeammateRuntime(
            spec=spec,
            llm=MockLlm([single_text_turn("done")]),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        rt = PersistentRuntime(teammate, bus)

        await rt.start("do work")
        await asyncio.sleep(0.3)
        assert teammate.alive
        assert rt.alive

        await rt.stop()
        assert not teammate.alive
        assert not rt.alive
        assert teammate.status == "shutdown"
        assert rt.status is RuntimeStatus.SHUTDOWN
