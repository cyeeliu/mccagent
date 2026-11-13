"""InteractiveRuntime / RuntimeCoordinator / RuntimeFactory 测试。

Phase 1 验收标准：
- InteractiveRuntime 多轮交互、并发保护、cancel、stop。
- RuntimeCoordinator 注册、重复 ID、查询、stop_all、异常聚合。
- RuntimeFactory Runtime 类型和配置映射、ID 唯一性。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import AgentEvent, TextDelta, TurnEnd, Usage
from mcc_harness import (
    AgentRuntime,
    EphemeralRuntime,
    InteractiveRuntime,
    MessageBus,
    PersistentRuntime,
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


# -- InteractiveRuntime 测试 --


class TestInteractiveRuntime:
    def test_initial_state(self):
        agent = FakeAgent()
        rt = InteractiveRuntime(agent, runtime_id="rt-1", session_id="s1")
        assert rt.runtime_id == "rt-1"
        assert rt.session_id == "s1"
        assert rt.status is RuntimeStatus.CREATED
        assert not rt.alive

    def test_auto_runtime_id(self):
        rt = InteractiveRuntime(FakeAgent())
        assert rt.runtime_id.startswith("interactive::")

    def test_agent_id_defaults_to_runtime_id(self):
        rt = InteractiveRuntime(FakeAgent(), runtime_id="rt-x")
        assert rt.agent_id == "rt-x"

    def test_explicit_agent_id(self):
        rt = InteractiveRuntime(
            FakeAgent(), runtime_id="rt-x", agent_id="agent-42",
        )
        assert rt.agent_id == "agent-42"

    async def test_start_transitions_to_idle(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        assert rt.status is RuntimeStatus.IDLE
        assert rt.alive

    async def test_start_with_task_is_accepted(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start("initial prompt")
        assert rt.status is RuntimeStatus.IDLE

    async def test_double_start_raises(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        with pytest.raises(RuntimeError, match="already started"):
            await rt.start()

    async def test_start_after_close_raises(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.stop()
        with pytest.raises(RuntimeError, match="closed"):
            await rt.start()

    async def test_run_turn_yields_events(self):
        events = [TextDelta(text="a"), TextDelta(text="b"), TurnEnd(cost_usd=0.0)]
        rt = InteractiveRuntime(FakeAgent(events=events))
        await rt.start()

        collected: list[AgentEvent] = []
        async for ev in rt.run_turn("hello"):
            collected.append(ev)

        assert len(collected) == 3
        assert isinstance(collected[0], TextDelta)
        assert collected[0].text == "a"
        assert rt.status is RuntimeStatus.IDLE

    async def test_run_turn_drives_agent(self):
        agent = FakeAgent()
        rt = InteractiveRuntime(agent)
        await rt.start()

        async for _ in rt.run_turn("test prompt"):
            pass

        assert agent.run_count == 1
        assert agent.last_prompt == "test prompt"

    async def test_run_turn_after_close_raises(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.stop()
        with pytest.raises(RuntimeError, match="closed"):
            async for _ in rt.run_turn("x"):
                pass

    async def test_concurrent_turns_blocked(self):
        rt = InteractiveRuntime(FakeAgent(delay=0.05))
        await rt.start()

        async def collect() -> int:
            count = 0
            async for _ in rt.run_turn("p"):
                count += 1
            return count

        results = await asyncio.gather(collect(), collect())
        assert all(r > 0 for r in results)

    async def test_stop_transitions_to_shutdown(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        await rt.stop()
        assert rt.status is RuntimeStatus.SHUTDOWN
        assert not rt.alive

    async def test_stop_is_idempotent(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        await rt.stop()
        await rt.stop()
        assert rt.status is RuntimeStatus.SHUTDOWN

    async def test_cancel_during_turn(self):
        agent = SlowAgent()
        rt = InteractiveRuntime(agent)
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
        assert rt.status is RuntimeStatus.IDLE

    async def test_wait_done_after_stop(self):
        usage = Usage(input_tokens=10, output_tokens=5)
        agent = FakeAgent(usage=usage, cost=0.01)
        rt = InteractiveRuntime(agent, runtime_id="rt-w", agent_id="a-w")
        await rt.start()

        async for _ in rt.run_turn("p"):
            pass

        await rt.stop()
        result = await rt.wait_done()
        assert result.is_terminal
        assert result.status is RuntimeStatus.SHUTDOWN
        assert result.runtime_id == "rt-w"
        assert result.agent_id == "a-w"
        assert result.usage == usage
        assert result.cost_usd == 0.01

    async def test_wait_done_not_closed_raises_timeout(self):
        from mcc_harness.runtime.base import WaitTimeoutError

        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        with pytest.raises(WaitTimeoutError):
            await rt.wait_done(timeout=1.0)

    async def test_wait_done_not_closed_no_timeout(self):
        rt = InteractiveRuntime(FakeAgent())
        await rt.start()
        result = await rt.wait_done()
        assert result.status is RuntimeStatus.IDLE

    async def test_config_mode_interactive(self):
        rt = InteractiveRuntime(FakeAgent(), agent_id="a1", session_id="s1")
        cfg = rt.config
        assert cfg.mode is RuntimeMode.INTERACTIVE
        assert cfg.agent_id == "a1"
        assert cfg.session_id == "s1"

    async def test_satisfies_protocol(self):
        rt = InteractiveRuntime(FakeAgent())
        assert isinstance(rt, AgentRuntime)

    async def test_multiple_turns(self):
        agent = FakeAgent()
        rt = InteractiveRuntime(agent)
        await rt.start()

        for i in range(3):
            async for _ in rt.run_turn(f"prompt-{i}"):
                pass

        assert agent.run_count == 3
        assert agent.last_prompt == "prompt-2"
        assert rt.status is RuntimeStatus.IDLE

    async def test_failing_agent_emits_error_event(self):
        from mcc_core import Error as ErrorEvent

        rt = InteractiveRuntime(FailingAgent())
        await rt.start()

        collected: list[AgentEvent] = []
        async for ev in rt.run_turn("trigger"):
            collected.append(ev)

        assert any(isinstance(ev, ErrorEvent) for ev in collected)
        assert rt.status is RuntimeStatus.IDLE


# -- RuntimeCoordinator 测试 --


class TestRuntimeCoordinator:
    def test_register_and_get(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="c-1")
        coord.register(rt)
        assert coord.get("c-1") is rt

    def test_register_duplicate_raises(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntime(FakeAgent(), runtime_id="dup")
        rt2 = InteractiveRuntime(FakeAgent(), runtime_id="dup")
        coord.register(rt1)
        with pytest.raises(RuntimeError, match="duplicate"):
            coord.register(rt2)

    def test_unregister(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="u-1")
        coord.register(rt)
        removed = coord.unregister("u-1")
        assert removed is rt
        assert coord.get("u-1") is None

    def test_unregister_nonexistent(self):
        coord = RuntimeCoordinator()
        assert coord.unregister("nope") is None

    def test_list_all(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntime(FakeAgent(), runtime_id="la-1")
        rt2 = InteractiveRuntime(FakeAgent(), runtime_id="la-2")
        coord.register(rt1)
        coord.register(rt2)
        all_rts = coord.list_all()
        assert len(all_rts) == 2

    def test_list_active(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="ac-1")
        coord.register(rt)
        assert len(coord.list_active()) == 0

    async def test_list_active_after_start(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="ac-2")
        coord.register(rt)
        await rt.start()
        assert len(coord.list_active()) == 1
        await rt.stop()
        assert len(coord.list_active()) == 0

    def test_snapshot(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="sn-1")
        coord.register(rt)
        snaps = coord.snapshot()
        assert len(snaps) == 1
        assert isinstance(snaps[0], RuntimeSnapshot)
        assert snaps[0].runtime_id == "sn-1"
        assert snaps[0].status is RuntimeStatus.CREATED
        assert snaps[0].alive is False

    def test_count(self):
        coord = RuntimeCoordinator()
        assert coord.count == 0
        coord.register(InteractiveRuntime(FakeAgent(), runtime_id="c-1"))
        assert coord.count == 1
        coord.register(InteractiveRuntime(FakeAgent(), runtime_id="c-2"))
        assert coord.count == 2

    async def test_active_count(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="ac-3")
        coord.register(rt)
        assert coord.active_count == 0
        await rt.start()
        assert coord.active_count == 1
        await rt.stop()
        assert coord.active_count == 0

    async def test_stop_all(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntime(FakeAgent(), runtime_id="sa-1")
        rt2 = InteractiveRuntime(FakeAgent(), runtime_id="sa-2")
        coord.register(rt1)
        coord.register(rt2)
        await rt1.start()
        await rt2.start()

        report = await coord.stop_all()
        assert isinstance(report, ShutdownReport)
        assert set(report.stopped) == {"sa-1", "sa-2"}
        assert not report.has_errors
        assert report.total == 2
        assert rt1.status is RuntimeStatus.SHUTDOWN
        assert rt2.status is RuntimeStatus.SHUTDOWN

    async def test_stop_all_idempotent(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(FakeAgent(), runtime_id="si-1")
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
                self._status = RuntimeStatus.RUNNING

            @property
            def runtime_id(self) -> str:
                return self._rid

            @property
            def status(self) -> RuntimeStatus:
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
                    status=RuntimeStatus.FAILED,
                    summary="broken",
                )

        coord = RuntimeCoordinator()
        good = InteractiveRuntime(FakeAgent(), runtime_id="g-1")
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
        rt = InteractiveRuntime(FakeAgent(), runtime_id="so-1")
        coord.register(rt)
        await rt.start()
        await coord.stop_one("so-1")
        assert rt.status is RuntimeStatus.SHUTDOWN

    async def test_stop_one_nonexistent(self):
        coord = RuntimeCoordinator()
        await coord.stop_one("nope")

    async def test_cancel_all(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntime(SlowAgent(), runtime_id="ca-1")
        coord.register(rt)
        await rt.start()

        async def run() -> None:
            async for _ in rt.run_turn("long"):
                pass

        task = asyncio.create_task(run())
        await asyncio.sleep(0.05)
        await coord.cancel_all()
        await task
        assert rt.status is RuntimeStatus.IDLE

    async def test_clear_terminal(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntime(FakeAgent(), runtime_id="ct-1")
        rt2 = InteractiveRuntime(FakeAgent(), runtime_id="ct-2")
        coord.register(rt1)
        coord.register(rt2)
        await rt1.start()
        await rt1.stop()

        removed = coord.clear_terminal()
        assert "ct-1" in removed
        assert "ct-2" not in removed
        assert coord.get("ct-1") is None
        assert coord.get("ct-2") is not None


# -- RuntimeFactory 测试 --


class TestRuntimeFactory:
    def test_create_interactive(self):
        factory = RuntimeFactory()
        agent = FakeAgent()
        rt = factory.create_interactive(
            agent, session_id="s1", agent_id="a1",
        )
        assert isinstance(rt, InteractiveRuntime)
        assert rt.session_id == "s1"
        assert rt.agent_id == "a1"
        assert rt.status is RuntimeStatus.CREATED

    def test_create_interactive_auto_id(self):
        factory = RuntimeFactory()
        rt1 = factory.create_interactive(FakeAgent())
        rt2 = factory.create_interactive(FakeAgent())
        assert rt1.runtime_id != rt2.runtime_id
        assert rt1.runtime_id.startswith("interactive::")

    def test_create_interactive_explicit_id(self):
        factory = RuntimeFactory()
        rt = factory.create_interactive(
            FakeAgent(), runtime_id="custom-rt",
        )
        assert rt.runtime_id == "custom-rt"

    def test_create_ephemeral(self, tmp_path):
        factory = RuntimeFactory()
        registry = default_registry()
        runner = SubagentRunner(MockLlm([single_text_turn("ok")]), registry)
        spec = default_presets()["code_explorer"]
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="fac", depth=0)
        rt = factory.create_ephemeral(runner, spec, ctx)
        assert isinstance(rt, EphemeralRuntime)
        assert rt.status is RuntimeStatus.CREATED

    def test_create_persistent(self, tmp_path):
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
        rt = factory.create_persistent(teammate, bus)
        assert isinstance(rt, PersistentRuntime)
        assert rt.status is RuntimeStatus.CREATED

    def test_factory_does_not_cache(self):
        factory = RuntimeFactory()
        agent = FakeAgent()
        rt1 = factory.create_interactive(agent)
        rt2 = factory.create_interactive(agent)
        assert rt1 is not rt2
