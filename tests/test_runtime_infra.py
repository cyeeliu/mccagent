"""InteractiveRuntimeV2 / RuntimeCoordinator / RuntimeFactory 测试。

Phase 1 验收标准：
- InteractiveRuntimeV2 多轮交互、并发保护、cancel、stop。
- RuntimeCoordinator 注册、重复 ID、查询、stop_all、异常聚合。
- RuntimeFactory Runtime 类型和配置映射、ID 唯一性。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import AgentEvent, TextDelta, TurnEnd, Usage
from mcc_core.runtime import RuntimeStatus as V2RuntimeStatus
from mcc_core.runtime.protocols import AgentRuntime
from mcc_harness import (
    InteractiveRuntimeV2,
    MessageBus,
    PersistentRuntimeV2,
    PlanStore,
    RuntimeConfig,
    RuntimeCoordinator,
    RuntimeFactory,
    RuntimeMode,
    RuntimeResult,
    RuntimeSnapshot,
    RuntimeStatus,
    ShutdownReport,
    SubagentRunner,
    SubagentSpec,
    TeammateRuntime,
    TeammateSpec,
    default_presets,
)
from mcc_harness.agent import AgentRun
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry, default_registry


class FakeAgent:
    """模拟 AgentLoop.run_streaming 的 push 模型行为。"""

    def __init__(
        self,
        events: list[AgentEvent] | None = None,
        delay: float = 0.0,
        usage: Usage | None = None,
        cost: float = 0.0,
    ) -> None:
        self._events = events or [TextDelta(text="hello"), TurnEnd(cost_usd=0.0)]
        self._delay = delay
        self._usage = usage or Usage()
        self._cost = cost
        self.run_count = 0
        self.last_prompt: str | None = None

    async def run_streaming(
        self,
        prompt: str,
        tx: asyncio.Queue[AgentEvent],
        cancel_event: asyncio.Event | None = None,
    ) -> AgentRun:
        self.run_count += 1
        self.last_prompt = prompt
        for ev in self._events:
            if cancel_event is not None and cancel_event.is_set():
                break
            await tx.put(ev)
            if self._delay > 0:
                await asyncio.sleep(self._delay)
        return AgentRun(
            final_text="done",
            iterations=1,
            total_usage=self._usage,
            cost_usd=self._cost,
        )


class SlowAgent(FakeAgent):
    """模拟长时间运行的 Agent，用于取消测试。"""

    def __init__(self) -> None:
        super().__init__(delay=0.05)

    async def run_streaming(
        self,
        prompt: str,
        tx: asyncio.Queue[AgentEvent],
        cancel_event: asyncio.Event | None = None,
    ) -> AgentRun:
        for i in range(100):
            if cancel_event is not None and cancel_event.is_set():
                return AgentRun(
                    final_text="cancelled",
                    iterations=i,
                    total_usage=Usage(),
                    cost_usd=0.0,
                )
            await tx.put(TextDelta(text=f"chunk {i}"))
            await asyncio.sleep(self._delay)
        return AgentRun(
            final_text="done",
            iterations=100,
            total_usage=Usage(),
            cost_usd=0.0,
        )


class FailingAgent(FakeAgent):
    """模拟运行时抛异常的 Agent。"""

    async def run_streaming(
        self,
        prompt: str,
        tx: asyncio.Queue[AgentEvent],
        cancel_event: asyncio.Event | None = None,
    ) -> AgentRun:
        raise RuntimeError("agent failed")


# -- InteractiveRuntimeV2 测试 --


class TestInteractiveRuntime:
    def test_initial_state(self):
        agent = FakeAgent()
        rt = InteractiveRuntimeV2(agent, runtime_id="rt-1", session_id="s1")
        assert rt.runtime_id == "rt-1"
        assert rt.session_id == "s1"
        assert rt.status is V2RuntimeStatus.CREATED
        assert not rt.alive

    def test_auto_runtime_id(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        assert rt.runtime_id.startswith("interactive::")

    def test_agent_id_defaults_to_runtime_id(self):
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="rt-x")
        assert rt.agent_id == "rt-x"

    def test_explicit_agent_id(self):
        rt = InteractiveRuntimeV2(
            FakeAgent(), runtime_id="rt-x", agent_id="agent-42",
        )
        assert rt.agent_id == "agent-42"

    async def test_start_transitions_to_ready(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.start()
        assert rt.status is V2RuntimeStatus.READY
        assert rt.alive

    async def test_start_is_idempotent_after_ready(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.start()
        with pytest.raises(RuntimeError, match="already started"):
            await rt.start()

    async def test_start_after_close_raises(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.stop()
        with pytest.raises(RuntimeError, match="closed"):
            await rt.start()

    async def test_run_turn_yields_events(self):
        events = [TextDelta(text="a"), TextDelta(text="b"), TurnEnd(cost_usd=0.0)]
        rt = InteractiveRuntimeV2(FakeAgent(events=events))
        await rt.start()

        collected: list[AgentEvent] = []
        async for ev in rt.run_turn("hello"):
            collected.append(ev)

        assert len(collected) == 3
        assert isinstance(collected[0], TextDelta)
        assert collected[0].text == "a"
        assert rt.status is V2RuntimeStatus.READY

    async def test_run_turn_drives_agent(self):
        agent = FakeAgent()
        rt = InteractiveRuntimeV2(agent)
        await rt.start()

        async for _ in rt.run_turn("test prompt"):
            pass

        assert agent.run_count == 1
        assert agent.last_prompt == "test prompt"

    async def test_run_turn_after_close_raises(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.stop()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in rt.run_turn("x"):
                pass

    async def test_concurrent_turns_blocked(self):
        rt = InteractiveRuntimeV2(FakeAgent(delay=0.05))
        await rt.start()

        async def collect() -> int:
            count = 0
            async for _ in rt.run_turn("p"):
                count += 1
            return count

        results = await asyncio.gather(collect(), collect())
        assert all(r > 0 for r in results)

    async def test_stop_transitions_to_stopped(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.start()
        await rt.stop()
        assert rt.status is V2RuntimeStatus.STOPPED
        assert not rt.alive

    async def test_stop_is_idempotent(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.start()
        await rt.stop()
        await rt.stop()
        assert rt.status is V2RuntimeStatus.STOPPED

    async def test_cancel_during_turn(self):
        agent = SlowAgent()
        rt = InteractiveRuntimeV2(agent)
        await rt.start()

        async def run() -> list[AgentEvent]:
            collected: list[AgentEvent] = []
            async for ev in rt.run_turn("long"):
                collected.append(ev)
            return collected

        task = asyncio.create_task(run())
        await asyncio.sleep(0.1)
        rt.cancel()
        collected = await task
        assert len(collected) < 100
        assert rt.status is V2RuntimeStatus.READY

    async def test_wait_done_after_stop(self):
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="rt-w", agent_id="a-w")
        await rt.start()

        async for _ in rt.run_turn("p"):
            pass

        await rt.stop()
        status, _ = await rt.wait_done()
        assert status is V2RuntimeStatus.STOPPED

    async def test_wait_done_not_closed_returns_running_status(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        await rt.start()
        status, _ = await rt.wait_done(timeout=1.0)
        assert status is V2RuntimeStatus.READY

    async def test_wait_done_not_closed_no_timeout(self):
        rt = InteractiveRuntimeV2(FakeAgent(delay=0.05))
        await rt.start()

        async def run_and_stop() -> None:
            async for _ in rt.run_turn("p"):
                pass
            await rt.stop()

        asyncio.create_task(run_and_stop())
        await asyncio.sleep(0.01)
        status, _ = await rt.wait_done()
        assert status in (V2RuntimeStatus.READY, V2RuntimeStatus.STOPPED)

    async def test_session_and_agent_id_preserved(self):
        rt = InteractiveRuntimeV2(FakeAgent(), agent_id="a1", session_id="s1")
        assert rt.agent_id == "a1"
        assert rt.session_id == "s1"

    async def test_satisfies_protocol(self):
        rt = InteractiveRuntimeV2(FakeAgent())
        assert isinstance(rt, AgentRuntime)

    async def test_multiple_turns(self):
        agent = FakeAgent()
        rt = InteractiveRuntimeV2(agent)
        await rt.start()

        for i in range(3):
            async for _ in rt.run_turn(f"prompt-{i}"):
                pass

        assert agent.run_count == 3
        assert agent.last_prompt == "prompt-2"
        assert rt.status is V2RuntimeStatus.READY

    async def test_failing_agent_emits_error_event(self):
        from mcc_core import Error as ErrorEvent

        rt = InteractiveRuntimeV2(FailingAgent())
        await rt.start()

        collected: list[AgentEvent] = []
        async for ev in rt.run_turn("trigger"):
            collected.append(ev)

        assert any(isinstance(ev, ErrorEvent) for ev in collected)
        assert rt.status is V2RuntimeStatus.READY


# -- RuntimeCoordinator 测试 --


class TestRuntimeCoordinator:
    def test_register_and_get(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="c-1")
        coord.register(rt)
        assert coord.get("c-1") is rt

    def test_register_duplicate_raises(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="dup")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="dup")
        coord.register(rt1)
        with pytest.raises(RuntimeError, match="duplicate"):
            coord.register(rt2)

    def test_unregister(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="u-1")
        coord.register(rt)
        removed = coord.unregister("u-1")
        assert removed is rt
        assert coord.get("u-1") is None

    def test_unregister_nonexistent(self):
        coord = RuntimeCoordinator()
        assert coord.unregister("nope") is None

    def test_list_all(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="la-1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="la-2")
        coord.register(rt1)
        coord.register(rt2)
        all_rts = coord.list_all()
        assert len(all_rts) == 2

    def test_list_active(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ac-1")
        coord.register(rt)
        assert len(coord.list_active()) == 0

    async def test_list_active_after_start(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ac-2")
        coord.register(rt)
        await rt.start()
        assert len(coord.list_active()) == 1
        await rt.stop()
        assert len(coord.list_active()) == 0

    def test_snapshot(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="sn-1")
        coord.register(rt)
        snaps = coord.snapshot()
        assert len(snaps) == 1
        assert isinstance(snaps[0], RuntimeSnapshot)
        assert snaps[0].runtime_id == "sn-1"
        assert snaps[0].status is V2RuntimeStatus.CREATED
        assert snaps[0].alive is False

    def test_count(self):
        coord = RuntimeCoordinator()
        assert coord.count == 0
        coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id="c-1"))
        assert coord.count == 1
        coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id="c-2"))
        assert coord.count == 2

    async def test_active_count(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ac-3")
        coord.register(rt)
        assert coord.active_count == 0
        await rt.start()
        assert coord.active_count == 1
        await rt.stop()
        assert coord.active_count == 0

    async def test_stop_all(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="sa-1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="sa-2")
        coord.register(rt1)
        coord.register(rt2)
        await rt1.start()
        await rt2.start()

        report = await coord.stop_all()
        assert isinstance(report, ShutdownReport)
        assert set(report.stopped) == {"sa-1", "sa-2"}
        assert not report.has_errors
        assert report.total == 2
        assert rt1.status is V2RuntimeStatus.STOPPED
        assert rt2.status is V2RuntimeStatus.STOPPED

    async def test_stop_all_idempotent(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="si-1")
        coord.register(rt)
        await rt.start()

        await coord.stop_all()
        report = await coord.stop_all()
        assert report.total == 0

    async def test_stop_all_empty(self):
        coord = RuntimeCoordinator()
        report = await coord.stop_all()
        assert report.total == 0

    async def test_stop_all_aggregates_errors(self):
        class BrokenRuntime:
            def __init__(self) -> None:
                self._rid = "br-1"
                self._status = V2RuntimeStatus.STARTING

            @property
            def runtime_id(self) -> str:
                return self._rid

            @property
            def status(self) -> V2RuntimeStatus:
                return self._status

            @property
            def alive(self) -> bool:
                return True

            async def start(self, task: str = "", *, task_id=None) -> None:
                pass

            async def stop(self, *, timeout: float = 5.0) -> None:
                raise RuntimeError("stop failed")

            def cancel(self) -> None:
                pass

            async def wait_done(self, *, timeout=None) -> RuntimeResult:
                return RuntimeResult(
                    runtime_id=self._rid,
                    status=V2RuntimeStatus.FAILED,
                    summary="broken",
                )

        coord = RuntimeCoordinator()
        good = InteractiveRuntimeV2(FakeAgent(), runtime_id="g-1")
        broken = BrokenRuntime()
        coord.register(good)
        coord.register(broken)
        await good.start()

        report = await coord.stop_all()
        assert "g-1" in report.stopped
        assert "br-1" in report.failed
        assert report.failed["br-1"] == "stop failed"
        assert report.has_errors

    async def test_stop_one(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="so-1")
        coord.register(rt)
        await rt.start()
        await coord.stop_one("so-1")
        assert rt.status is V2RuntimeStatus.STOPPED

    async def test_stop_one_nonexistent(self):
        coord = RuntimeCoordinator()
        await coord.stop_one("nope")

    async def test_cancel_all(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(SlowAgent(), runtime_id="ca-1")
        coord.register(rt)
        await rt.start()

        async def run() -> None:
            async for _ in rt.run_turn("long"):
                pass

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        await coord.cancel_all()
        await task
        assert rt.status is V2RuntimeStatus.READY

    async def test_clear_terminal(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="ct-1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="ct-2")
        coord.register(rt1)
        coord.register(rt2)
        await rt1.start()
        await rt1.stop()

        removed = coord.clear_terminal()
        assert "ct-1" in removed
        assert "ct-2" not in removed
        assert coord.get("ct-1") is None
        assert coord.get("ct-2") is not None

    async def test_register_after_stop_all_rejected(self):
        from mcc_harness import CoordinatorClosedError

        coord = RuntimeCoordinator()
        await coord.stop_all()
        with pytest.raises(CoordinatorClosedError, match="closed"):
            coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id="late"))

    async def test_clear_terminal_includes_stopped(self):
        coord = RuntimeCoordinator()

        class StoppedRuntime:
            def __init__(self) -> None:
                self._rid = "be-1"

            @property
            def runtime_id(self) -> str:
                return self._rid

            @property
            def status(self) -> V2RuntimeStatus:
                return V2RuntimeStatus.STOPPED

            @property
            def alive(self) -> bool:
                return False

            async def start(self, task: str = "", *, task_id=None) -> None:
                pass

            async def stop(self, *, timeout: float = 5.0) -> None:
                pass

            def cancel(self) -> None:
                pass

            async def wait_done(self, *, timeout=None) -> RuntimeResult:
                return RuntimeResult(
                    runtime_id=self._rid,
                    status=V2RuntimeStatus.STOPPED,
                    summary="stopped",
                )

        rt = StoppedRuntime()
        coord.register(rt)
        removed = coord.clear_terminal()
        assert "be-1" in removed
        assert coord.get("be-1") is None

    async def test_register_during_stop_all_rejected(self):
        """P1-04: stop_all 执行期间 register 被拒绝。"""
        from mcc_harness import CoordinatorClosedError

        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="r-1")
        coord.register(rt1)
        await rt1.start()

        task = asyncio.create_task(coord.stop_all())
        await asyncio.sleep(0.01)

        with pytest.raises(CoordinatorClosedError, match="closed"):
            coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id="r-2"))

        await task
        assert coord.get("r-2") is None

    async def test_stop_all_then_register_always_rejected(self):
        """P1-04: stop_all 完成后多次 register 均被拒绝。"""
        from mcc_harness import CoordinatorClosedError

        coord = RuntimeCoordinator()
        await coord.stop_all()

        for i in range(3):
            with pytest.raises(CoordinatorClosedError):
                coord.register(
                    InteractiveRuntimeV2(FakeAgent(), runtime_id=f"late-{i}")
                )


# -- RuntimeFactory 测试 --


class TestRuntimeFactory:
    def test_create_interactive_v2_basic(self):
        factory = RuntimeFactory()
        agent = FakeAgent()
        rt = factory.create_interactive_v2(
            agent, session_id="s1", agent_id="a1",
        )
        assert isinstance(rt, InteractiveRuntimeV2)
        assert rt.session_id == "s1"
        assert rt.agent_id == "a1"
        assert rt.status is V2RuntimeStatus.CREATED

    def test_create_interactive_v2_auto_id(self):
        factory = RuntimeFactory()
        rt1 = factory.create_interactive_v2(FakeAgent())
        rt2 = factory.create_interactive_v2(FakeAgent())
        assert rt1.runtime_id != rt2.runtime_id
        assert rt1.runtime_id.startswith("interactive::")

    def test_create_interactive_v2_explicit_id(self):
        factory = RuntimeFactory()
        rt = factory.create_interactive_v2(
            FakeAgent(), runtime_id="custom-rt",
        )
        assert rt.runtime_id == "custom-rt"

    def test_create_persistent_v2_basic(self, tmp_path):
        factory = RuntimeFactory()
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="fac-worker", role="coder",
            system="test", model="claude-haiku-4-5",
        )
        teammate = TeammateRuntime(
            spec=spec,
            llm=MockLlm([single_text_turn("done")]),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        rt = factory.create_persistent_v2(teammate, bus)
        assert isinstance(rt, PersistentRuntimeV2)

    def test_factory_does_not_cache(self):
        factory = RuntimeFactory()
        agent = FakeAgent()
        rt1 = factory.create_interactive_v2(agent)
        rt2 = factory.create_interactive_v2(agent)
        assert rt1 is not rt2

    def test_create_interactive_v2(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        factory = RuntimeFactory()
        agent = FakeAgent()
        rt = factory.create_interactive_v2(agent, agent_id="a1")
        assert isinstance(rt, InteractiveRuntimeV2)
        assert rt.agent_id == "a1"

    def test_create_task(self):
        from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor
        from mcc_harness.runtime.task_runtime import TaskRuntime

        factory = RuntimeFactory()
        executor = AgentLoopExecutor(FakeAgent(), agent_id="t1")
        rt = factory.create_task(executor, agent_id="t1")
        assert isinstance(rt, TaskRuntime)
        assert rt.agent_id == "t1"

    def test_create_task_auto_id(self):
        from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor
        from mcc_harness.runtime.task_runtime import TaskRuntime

        factory = RuntimeFactory()
        executor = AgentLoopExecutor(FakeAgent())
        rt = factory.create_task(executor)
        assert isinstance(rt, TaskRuntime)
        assert rt.runtime_id.startswith("task::")

    def test_create_persistent_v2(self, tmp_path):
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        factory = RuntimeFactory()
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        spec = TeammateSpec(
            name="v2-worker", role="coder",
            system="test", model="claude-haiku-4-5",
        )
        teammate = TeammateRuntime(
            spec=spec,
            llm=MockLlm([single_text_turn("done")]),
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            base_dir=tmp_path,
        )
        rt = factory.create_persistent_v2(teammate, bus)
        assert isinstance(rt, PersistentRuntimeV2)

    def test_v2_factory_does_not_cache(self):
        factory = RuntimeFactory()
        agent = FakeAgent()
        rt1 = factory.create_interactive_v2(agent)
        rt2 = factory.create_interactive_v2(agent)
        assert rt1 is not rt2
