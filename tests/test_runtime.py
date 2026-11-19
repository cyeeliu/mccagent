"""统一 AgentRuntime 抽象测试：接口契约 + PersistentRuntimeV2 生命周期 + 兼容接口。

验证：
- PersistentRuntimeV2 通过新 Protocol 接口启动、停止和查询状态。
- submit_task / wait_task 兼容接口完成单任务执行闭环。
- 旧 API（SubagentRunner、TeammateRuntime）继续工作。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import ToolContext, Usage
from mcc_core.runtime import RuntimeStatus as V2RuntimeStatus
from mcc_core.runtime.protocols import AgentRuntime
from mcc_harness import (
    LEAD_NAME,
    MessageBus,
    PersistentRuntimeV2,
    PlanStore,
    RuntimeResult,
    RuntimeStatus,
    SubagentResult,
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


def _make_persistent(
    tmp_path: Path,
    llm=None,
    name: str = "worker",
) -> PersistentRuntimeV2:
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    registry = ToolRegistry()
    spec = TeammateSpec(
        name=name,
        role="coder",
        system="test",
        model="claude-haiku-4-5",
    )
    rt = TeammateRuntime(
        spec=spec,
        llm=llm or MockLlm([single_text_turn("done")]),
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        base_dir=tmp_path,
    )
    return PersistentRuntimeV2.from_teammate(rt, bus)


# -- 接口契约测试 --


class TestRuntimeContract:
    def test_v2_runtime_status_values(self):
        assert V2RuntimeStatus.CREATED == "created"
        assert V2RuntimeStatus.READY == "ready"
        assert V2RuntimeStatus.STOPPED == "stopped"

    @pytest.mark.parametrize(
        "status,expected",
        [
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
        ],
    )
    def test_result_is_terminal(self, status, expected):
        r = RuntimeResult(runtime_id="x", status=status, summary="")
        assert r.is_terminal is expected

    def test_persistent_satisfies_protocol(self, tmp_path):
        rt = _make_persistent(tmp_path)
        assert isinstance(rt, AgentRuntime)


# -- PersistentRuntimeV2 生命周期测试 --


class TestPersistentRuntimeV2:
    async def test_start_and_status(self, tmp_path):
        rt = _make_persistent(tmp_path)
        assert rt.status is V2RuntimeStatus.CREATED
        assert not rt.alive

        await rt.start()
        assert rt.alive
        await asyncio.sleep(0.3)
        assert rt.status in (
            V2RuntimeStatus.READY,
            V2RuntimeStatus.STARTING,
            V2RuntimeStatus.PAUSED,
        )
        await rt.stop()
        assert not rt.alive

    async def test_stop_transitions_to_stopped(self, tmp_path):
        rt = _make_persistent(tmp_path)
        await rt.start()
        await asyncio.sleep(0.1)
        await rt.stop()
        assert rt.status is V2RuntimeStatus.STOPPED

    async def test_cancel_sends_signal(self, tmp_path):
        rt = _make_persistent(tmp_path)
        await rt.start()
        await asyncio.sleep(0.1)
        rt.cancel()
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_pause_and_resume(self, tmp_path):
        llm = MockLlm(
            [
                single_text_turn("first"),
                single_text_turn("second"),
            ]
        )
        rt = _make_persistent(tmp_path, llm=llm, name="pausable")
        await rt.start()
        await asyncio.sleep(0.2)
        await rt.pause()
        await asyncio.sleep(0.1)
        await rt.resume()
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_inject_instruction(self, tmp_path):
        llm = MockLlm(
            [
                single_text_turn("first"),
                single_text_turn("second"),
            ]
        )
        rt = _make_persistent(tmp_path, llm=llm, name="injectable")
        await rt.start()
        await asyncio.sleep(0.1)
        rt.inject_instruction("extra guidance")
        await asyncio.sleep(0.2)
        await rt.stop()
        assert not rt.alive

    async def test_runtime_id_is_unique_uuid(self, tmp_path):
        rt = _make_persistent(tmp_path, name="my-worker")
        assert rt.runtime_id.startswith("persistent::")
        rt2 = _make_persistent(tmp_path, name="my-worker")
        assert rt.runtime_id != rt2.runtime_id

    async def test_cancel_when_not_alive_is_noop(self, tmp_path):
        rt = _make_persistent(tmp_path)
        rt.cancel()
        await rt.pause()
        await rt.resume()
        rt.inject_instruction("test")

    async def test_work_result_via_bus(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="bus-test",
            role="coder",
            system="test",
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
        rt = PersistentRuntimeV2.from_teammate(teammate, bus)
        await rt.start()
        await rt.submit_task("do the work")
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        assert any(m.kind == "work_done" for m in msgs)
        await rt.stop()

    async def test_stop_passes_timeout_to_shutdown(self, tmp_path):
        rt = _make_persistent(tmp_path)
        captured: list[float] = []
        original = rt.teammate.shutdown

        async def capture(*, timeout: float = 5.0) -> None:
            captured.append(timeout)
            await original(timeout=timeout)

        rt.teammate.shutdown = capture
        await rt.start()
        await asyncio.sleep(0.2)
        await rt.stop(timeout=3.0)
        assert captured == [3.0]

    async def test_stop_default_timeout_is_5(self, tmp_path):
        rt = _make_persistent(tmp_path)
        captured: list[float] = []
        original = rt.teammate.shutdown

        async def capture(*, timeout: float = 5.0) -> None:
            captured.append(timeout)
            await original(timeout=timeout)

        rt.teammate.shutdown = capture
        await rt.start()
        await asyncio.sleep(0.2)
        await rt.stop()
        assert captured == [5.0]

    async def test_submit_task_and_wait_task(self, tmp_path):
        llm = MockLlm(
            [
                single_text_turn("task result"),
            ]
        )
        rt = _make_persistent(tmp_path, llm=llm, name="submit-test")
        await rt.start()
        await asyncio.sleep(0.1)

        tid = await rt.submit_task("do something", task_id="t-001")
        assert tid == "t-001"
        result = await rt.wait_task(tid, timeout=5.0)
        assert result.status is RuntimeStatus.COMPLETED
        assert "task result" in result.summary
        await rt.stop()

    async def test_submit_multiple_tasks_sequentially(self, tmp_path):
        llm = MockLlm(
            [
                single_text_turn("first done"),
                single_text_turn("second done"),
                single_text_turn("third done"),
            ]
        )
        rt = _make_persistent(tmp_path, llm=llm, name="multi-submit")
        await rt.start()
        await asyncio.sleep(0.3)

        tid1 = await rt.submit_task("task one", task_id="t1")
        r1 = await rt.wait_task(tid1, timeout=5.0)
        assert r1.status is RuntimeStatus.COMPLETED

        tid2 = await rt.submit_task("task two", task_id="t2")
        r2 = await rt.wait_task(tid2, timeout=5.0)
        assert r2.status is RuntimeStatus.COMPLETED

        await rt.stop()

    async def test_wait_task_unknown_id_returns_failed(self, tmp_path):
        rt = _make_persistent(tmp_path, name="unknown-task")
        await rt.start()
        await asyncio.sleep(0.2)
        result = await rt.wait_task("nonexistent", timeout=1.0)
        assert result.status is RuntimeStatus.FAILED
        assert "unknown task_id" in result.summary
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
            name="compat",
            role="coder",
            system="test",
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

    async def test_persistent_v2_wraps_teammate_consistent(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="cons",
            role="coder",
            system="test",
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
        rt = PersistentRuntimeV2.from_teammate(teammate, bus)

        await rt.start()
        await asyncio.sleep(0.3)
        assert teammate.alive
        assert rt.alive

        await rt.stop()
        assert not teammate.alive
        assert not rt.alive
        assert rt.status is V2RuntimeStatus.STOPPED

    async def test_stop_timeout_raises_runtime_stop_timeout(self, tmp_path):
        from mcc_harness import RuntimeStopTimeoutError

        class HangingLlm:
            async def complete(self, req):
                await asyncio.sleep(100)
                raise RuntimeError("unreachable")

            async def stream(self, req):
                await asyncio.sleep(100)
                yield  # pragma: no cover

        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="hang",
            role="coder",
            system="test",
            model="claude-haiku-4-5",
        )
        teammate = TeammateRuntime(
            spec=spec,
            llm=HangingLlm(),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        rt = PersistentRuntimeV2.from_teammate(teammate, bus)
        await rt.start()
        await rt.submit_task("hang work", task_id="t-hang")
        await asyncio.sleep(0.3)
        with pytest.raises(RuntimeStopTimeoutError):
            await rt.stop(timeout=0.05)
        assert not rt.alive
