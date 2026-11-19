"""P1-12~P1-15 Coordinator/Application 生命周期完善回归测试。

覆盖：
- P1-12: Coordinator OPEN/CLOSING/CLOSED 状态枚举与并发事务。
- P1-13: 健康回调去重，每个 Runtime 独立监控状态。
- P1-14: Application ApplicationState 与 shutdown 第一阶段关闭 Admission。
- P1-15: 统一 ApplicationShutdownReport，总 timeout 作为整个 shutdown deadline。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcc_cli.application import (
    AgentApplication,
    ApplicationShutdownReport,
    ApplicationState,
)
from mcc_core.runtime.capabilities import RuntimeCapabilities
from mcc_core.runtime.health import HealthState, RuntimeHealth
from mcc_harness import (
    CoordinatorClosedError,
    CoordinatorState,
    InteractiveRuntimeV2,
    RuntimeCoordinator,
    RuntimeMonitorState,
    RuntimeStatus,
    ShutdownReport,
)


class FakeAgent:
    """最小 Agent 桩，满足 InteractiveRuntimeV2 构造。"""

    async def run(self, messages, *, tools=None, system=None, model=None):
        yield


class StubRuntime:
    """可控 Runtime 桩，支持健康检查与启停追踪。"""

    def __init__(
        self,
        rid: str,
        *,
        healthy: bool = True,
        slow_start: float = 0.0,
        agent_id: str | None = None,
    ) -> None:
        self._rid = rid
        self._agent_id = agent_id or rid
        self._healthy = healthy
        self._slow_start = slow_start
        self._status = RuntimeStatus.CREATED
        self._alive = False
        self.start_count = 0
        self.stop_count = 0

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=True)

    async def start(self, task: str = "", *, task_id=None) -> None:
        self.start_count += 1
        if self._slow_start > 0:
            await asyncio.sleep(self._slow_start)
        self._status = RuntimeStatus.RUNNING
        self._alive = True

    async def stop(self, *, timeout: float = 5.0) -> None:
        self.stop_count += 1
        self._status = RuntimeStatus.SHUTDOWN
        self._alive = False

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="done",
        )

    async def health_check(self) -> RuntimeHealth:
        state = HealthState.HEALTHY if self._healthy else HealthState.UNHEALTHY
        return RuntimeHealth(runtime_id=self._rid, state=state)

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy


class FlakyHealthRuntime(StubRuntime):
    """健康状态可动态切换的 Runtime 桩。"""

    def __init__(self, rid: str) -> None:
        super().__init__(rid)
        self._health_sequence: list[bool] = []
        self._check_index = 0

    def set_health_sequence(self, sequence: list[bool]) -> None:
        self._health_sequence = sequence
        self._check_index = 0

    async def health_check(self) -> RuntimeHealth:
        if self._check_index < len(self._health_sequence):
            self._healthy = self._health_sequence[self._check_index]
            self._check_index += 1
        state = HealthState.HEALTHY if self._healthy else HealthState.UNHEALTHY
        return RuntimeHealth(runtime_id=self._rid, state=state)


# ============================================================
# P1-12: Coordinator 状态枚举与并发事务
# ============================================================


class TestP1_12CoordinatorState:
    """Coordinator 引入 OPEN/CLOSING/CLOSED 状态枚举。"""

    def test_initial_state_is_open(self):
        coord = RuntimeCoordinator()
        assert coord.state is CoordinatorState.OPEN
        assert not coord.is_closed
        assert not coord.is_closing

    async def test_coordinator_state_open_to_closing_to_closed(self):
        """验证状态转换 OPEN → CLOSING → CLOSED。"""
        coord = RuntimeCoordinator()
        rt = StubRuntime("s1")
        await coord.register_and_start(rt)
        assert coord.state is CoordinatorState.OPEN

        await coord.begin_shutdown()
        assert coord.state is CoordinatorState.CLOSING
        assert coord.is_closing
        assert not coord.is_closed

        report = await coord.shutdown_all()
        assert isinstance(report, ShutdownReport)
        assert coord.state is CoordinatorState.CLOSED
        assert coord.is_closed
        assert coord.is_closing

    async def test_begin_shutdown_is_idempotent(self):
        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        assert coord.state is CoordinatorState.CLOSING
        await coord.begin_shutdown()
        assert coord.state is CoordinatorState.CLOSING

    async def test_register_rejected_when_closing(self):
        """CLOSING 状态下 register 抛 CoordinatorClosedError。"""
        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        assert coord.state is CoordinatorState.CLOSING
        rt = StubRuntime("r1")
        with pytest.raises(CoordinatorClosedError):
            coord.register(rt)

    async def test_register_rejected_when_closed(self):
        """CLOSED 状态下 register 抛 CoordinatorClosedError。"""
        coord = RuntimeCoordinator()
        await coord.shutdown_all()
        assert coord.state is CoordinatorState.CLOSED
        rt = StubRuntime("r1")
        with pytest.raises(CoordinatorClosedError):
            coord.register(rt)

    async def test_register_and_start_rejected_when_closing(self):
        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        rt = StubRuntime("r1")
        with pytest.raises(CoordinatorClosedError):
            await coord.register_and_start(rt)

    async def test_concurrent_register_and_shutdown(self):
        """并发 register 和 shutdown_all，新 Runtime 不被遗漏或被正确回滚。"""
        coord = RuntimeCoordinator()
        results: list[Any] = []

        async def do_shutdown():
            await asyncio.sleep(0.05)
            report = await coord.shutdown_all(timeout=2.0)
            results.append(report)

        async def do_register():
            rt = StubRuntime("race-1", slow_start=0.1)
            try:
                await coord.register_and_start(rt)
                results.append(("registered", rt))
            except CoordinatorClosedError:
                results.append(("rolled_back", rt))

        await asyncio.gather(do_shutdown(), do_register())

        report = next(r for r in results if isinstance(r, ShutdownReport))
        registrations = [r for r in results if isinstance(r, tuple)]
        assert coord.state is CoordinatorState.CLOSED

        for tag, rt in registrations:
            if tag == "registered":
                assert (
                    "race-1" in report.stopped
                    or "race-1" in report.failed
                    or "race-1" in report.timed_out
                ), "已注册的 Runtime 必须被 shutdown 处理"
            else:
                assert rt.start_count >= 1
                assert rt.stop_count >= 1, "回滚的 Runtime 必须被停止"

    async def test_register_and_start_rollback_on_shutdown(self):
        """register_and_start 在 shutdown 并发时回滚。"""
        coord = RuntimeCoordinator()

        async def do_shutdown():
            await asyncio.sleep(0.02)
            await coord.shutdown_all(timeout=2.0)

        async def do_register():
            rt = StubRuntime("rb-1", slow_start=0.1)
            with pytest.raises(CoordinatorClosedError):
                await coord.register_and_start(rt)
            assert rt.stop_count >= 1, "回滚必须停止刚启动的 Runtime"
            assert coord.get("rb-1") is None, "回滚后不应残留注册记录"

        await asyncio.gather(do_shutdown(), do_register())

    async def test_shutdown_all_idempotent_after_closed(self):
        coord = RuntimeCoordinator()
        await coord.shutdown_all()
        assert coord.state is CoordinatorState.CLOSED
        report = await coord.shutdown_all()
        assert report.stopped == []
        assert report.failed == {}
        assert report.timed_out == []


# ============================================================
# P1-13: 健康回调去重
# ============================================================


class TestP1_13HealthCallbackDedup:
    """健康回调去重，每个 Runtime 独立监控状态。"""

    async def test_health_callback_not_duplicated(self):
        """Runtime 持续 UNHEALTHY 时回调只触发一次。"""
        coord = RuntimeCoordinator()
        rt = StubRuntime("hd-1", healthy=False)
        await coord.register_and_start(rt)

        callback_count: list[int] = []

        async def on_unhealthy(rid: str, status: RuntimeStatus) -> None:
            callback_count.append(1)

        coord.on_exit(on_unhealthy)
        await coord.start_health_monitor(interval=0.02)
        await asyncio.sleep(0.15)
        await coord.stop_health_monitor()

        assert len(callback_count) == 1, "持续 UNHEALTHY 回调应只触发一次"
        await coord.shutdown_all()

    async def test_health_callback_recovery_resets(self):
        """Runtime 从 UNHEALTHY 恢复后再次 UNHEALTHY 时回调再次触发。"""
        coord = RuntimeCoordinator()
        rt = FlakyHealthRuntime("hd-2")
        rt.set_health_sequence([False, False, True, True, False, False])
        await coord.register_and_start(rt)

        callback_count: list[int] = []

        async def on_unhealthy(rid: str, status: RuntimeStatus) -> None:
            callback_count.append(1)

        coord.on_exit(on_unhealthy)
        await coord.start_health_monitor(interval=0.02)
        await asyncio.sleep(0.2)
        await coord.stop_health_monitor()

        assert len(callback_count) == 2, "恢复后再次故障应重新触发回调"
        await coord.shutdown_all()

    async def test_health_monitor_state_per_runtime(self):
        """每个 Runtime 有独立的监控状态。"""
        coord = RuntimeCoordinator()
        rt1 = StubRuntime("ms-1", healthy=False)
        rt2 = StubRuntime("ms-2", healthy=True)
        await coord.register_and_start(rt1)
        await coord.register_and_start(rt2)

        await coord.start_health_monitor(interval=0.02)
        await asyncio.sleep(0.08)

        assert "ms-1" in coord._monitor_states
        assert "ms-2" in coord._monitor_states
        state1 = coord._monitor_states["ms-1"]
        state2 = coord._monitor_states["ms-2"]
        assert isinstance(state1, RuntimeMonitorState)
        assert isinstance(state2, RuntimeMonitorState)
        assert state1.failure_reported is True
        assert state2.failure_reported is False
        assert state1.last_health == "unhealthy"
        assert state2.last_health == "healthy"

        await coord.stop_health_monitor()
        await coord.shutdown_all()

    async def test_health_recovery_logged(self):
        """Runtime 恢复健康时重置 failure_reported。"""
        coord = RuntimeCoordinator()
        rt = FlakyHealthRuntime("rc-1")
        rt.set_health_sequence([False, False, True, True])
        await coord.register_and_start(rt)

        await coord.start_health_monitor(interval=0.02)
        await asyncio.sleep(0.12)
        await coord.stop_health_monitor()

        state = coord._monitor_states["rc-1"]
        assert state.failure_reported is False
        assert state.last_health == "healthy"
        await coord.shutdown_all()

    async def test_multiple_runtimes_independent_callbacks(self):
        """多个 Runtime 同时 UNHEALTHY，各自只触发一次。"""
        coord = RuntimeCoordinator()
        rt1 = StubRuntime("mi-1", healthy=False)
        rt2 = StubRuntime("mi-2", healthy=False)
        await coord.register_and_start(rt1)
        await coord.register_and_start(rt2)

        calls: list[str] = []

        async def on_unhealthy(rid: str, status: RuntimeStatus) -> None:
            calls.append(rid)

        coord.on_exit(on_unhealthy)
        await coord.start_health_monitor(interval=0.02)
        await asyncio.sleep(0.12)
        await coord.stop_health_monitor()

        assert calls.count("mi-1") == 1
        assert calls.count("mi-2") == 1
        await coord.shutdown_all()


# ============================================================
# P1-14: Application ApplicationState 与 Admission 关闭
# ============================================================


class TestP1_14ApplicationState:
    """Application 引入 ApplicationState，shutdown 第一阶段关闭 Admission。"""

    def test_initial_state_is_running(self):
        app = AgentApplication()
        assert app.state is ApplicationState.RUNNING
        assert not app.is_stopping

    async def test_application_state_transitions(self):
        """RUNNING → STOPPING → STOPPED。"""
        app = AgentApplication()
        assert app.state is ApplicationState.RUNNING

        await app.shutdown()
        assert app.state is ApplicationState.STOPPED
        assert app.is_stopping

    async def test_shutdown_blocks_new_registration(self):
        """STOPPING 状态下 register 失败。"""
        app = AgentApplication()
        await app.shutdown()
        assert app.state is ApplicationState.STOPPED

        rt = StubRuntime("blk-1")
        with pytest.raises(CoordinatorClosedError):
            app.register_runtime(rt)

    async def test_shutdown_first_step_closes_admission(self):
        """shutdown 第一阶段即调用 coordinator.begin_shutdown。"""
        app = AgentApplication()
        coord = app.coordinator
        assert coord.state is CoordinatorState.OPEN

        await app.shutdown()
        assert coord.state is CoordinatorState.CLOSED

    async def test_concurrent_shutdown_waits_same_task(self):
        """并发 shutdown 调用等待同一 task。"""
        app = AgentApplication()
        rt = StubRuntime("cs-1", slow_start=0.0)
        await app.register_and_start(rt)

        report1, report2 = await asyncio.gather(
            app.shutdown(timeout=5.0),
            app.shutdown(timeout=5.0),
        )

        assert isinstance(report1, ApplicationShutdownReport)
        assert isinstance(report2, ApplicationShutdownReport)
        assert app.state is ApplicationState.STOPPED
        assert rt.stop_count >= 1

    async def test_shutdown_idempotent_returns_same_report(self):
        app = AgentApplication()
        report1 = await app.shutdown()
        report2 = await app.shutdown()
        assert report1 is report2

    async def test_state_ordering(self):
        """ApplicationState 支持有序比较。"""
        assert ApplicationState.CREATED < ApplicationState.RUNNING
        assert ApplicationState.RUNNING < ApplicationState.STOPPING
        assert ApplicationState.STOPPING < ApplicationState.STOPPED
        assert ApplicationState.STOPPING >= ApplicationState.STOPPING


# ============================================================
# P1-15: ApplicationShutdownReport 与总 timeout
# ============================================================


class TestP1_15ShutdownReport:
    """统一 ApplicationShutdownReport，总 timeout 作为整个 shutdown deadline。"""

    async def test_shutdown_returns_application_shutdown_report(self):
        app = AgentApplication()
        report = await app.shutdown()
        assert isinstance(report, ApplicationShutdownReport)

    async def test_shutdown_report_includes_all_steps(self):
        """ApplicationShutdownReport 包含所有步骤。"""
        app = AgentApplication()
        report = await app.shutdown()
        assert "step1_admission" in report.completed_steps
        assert "step3_cancel_all" in report.completed_steps
        assert "step4_runtimes" in report.completed_steps
        assert "step7_clear_terminal" in report.completed_steps
        assert report.elapsed_seconds >= 0.0

    async def test_shutdown_report_with_all_subsystems(self):
        """带 team_manager / mcp / event_source 的完整报告。"""

        class StubTeamManager:
            def __init__(self) -> None:
                self.shutdown_called = False

            async def shutdown_all(self) -> None:
                self.shutdown_called = True

            def list_active(self):
                return []

        class StubMcpManager:
            def __init__(self) -> None:
                self.closed = False

            async def close_all(self) -> None:
                self.closed = True

        class StubEventSource:
            def __init__(self) -> None:
                self.shutdown_called = False

            async def shutdown(self, *, timeout: float = 10.0) -> None:
                self.shutdown_called = True

        team = StubTeamManager()
        mcp = StubMcpManager()
        event = StubEventSource()
        app = AgentApplication(mcp_manager=mcp, event_source=event, team_manager=team)
        report = await app.shutdown(timeout=5.0)

        assert "step2_team_manager" in report.completed_steps
        assert "step5_mcp" in report.completed_steps
        assert "step6_event_source" in report.completed_steps
        assert team.shutdown_called
        assert mcp.closed
        assert event.shutdown_called

    async def test_shutdown_total_timeout(self):
        """总 timeout 作为整个 shutdown deadline。"""
        app = AgentApplication()

        class SlowTeamManager:
            async def shutdown_all(self) -> None:
                await asyncio.sleep(10.0)

            def list_active(self):
                return []

        app._team_manager = SlowTeamManager()
        report = await app.shutdown(timeout=0.1)
        assert isinstance(report, ApplicationShutdownReport)
        assert app.state is ApplicationState.STOPPED
        assert "step2_team_manager" in report.timed_out_steps or "overall" in report.timed_out_steps

    async def test_shutdown_step_failure_recorded(self):
        """子系统失败记录到 failed_steps。"""

        class FailingTeamManager:
            async def shutdown_all(self) -> None:
                raise RuntimeError("team shutdown boom")

            def list_active(self):
                return []

        app = AgentApplication(team_manager=FailingTeamManager())
        report = await app.shutdown(timeout=5.0)
        assert "step2_team_manager" in report.failed_steps
        assert "team shutdown boom" in report.failed_steps["step2_team_manager"]
        assert app.state is ApplicationState.STOPPED

    async def test_shutdown_mcp_failure_recorded(self):
        class FailingMcp:
            async def close_all(self) -> None:
                raise RuntimeError("mcp boom")

        app = AgentApplication(mcp_manager=FailingMcp())
        report = await app.shutdown(timeout=5.0)
        assert "step5_mcp" in report.failed_steps

    async def test_shutdown_event_source_failure_recorded(self):
        class FailingEventSource:
            async def shutdown(self, *, timeout: float = 10.0) -> None:
                raise RuntimeError("event boom")

        app = AgentApplication(event_source=FailingEventSource())
        report = await app.shutdown(timeout=5.0)
        assert "step6_event_source" in report.failed_steps

    async def test_shutdown_report_has_errors_property(self):
        report = ApplicationShutdownReport()
        assert not report.has_errors
        report.failed_steps["x"] = "err"
        assert report.has_errors
        report2 = ApplicationShutdownReport()
        report2.timed_out_steps.append("y")
        assert report2.has_errors

    async def test_shutdown_report_stored_as_attribute(self):
        app = AgentApplication()
        report = await app.shutdown()
        assert app.shutdown_report is report

    async def test_remaining_time_prevents_long_step(self):
        """前一步消耗时间后，后续步骤使用剩余时间而非完整 timeout。"""

        class SlowThenFastTeam:
            def __init__(self) -> None:
                self.delay = 0.3

            async def shutdown_all(self) -> None:
                await asyncio.sleep(self.delay)

            def list_active(self):
                return []

        app = AgentApplication(team_manager=SlowThenFastTeam())
        report = await app.shutdown(timeout=0.2)
        assert app.state is ApplicationState.STOPPED
        assert isinstance(report, ApplicationShutdownReport)


# ============================================================
# 集成验证：真实 InteractiveRuntimeV2
# ============================================================


class TestP1LifecycleIntegration:
    """使用真实 InteractiveRuntimeV2 的集成验证。"""

    async def test_real_runtime_shutdown_state_transitions(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="int-1", agent_id="a1")
        await app.register_and_start(rt)
        assert app.state is ApplicationState.RUNNING

        report = await app.shutdown()
        assert app.state is ApplicationState.STOPPED
        assert app.coordinator.state is CoordinatorState.CLOSED
        assert not rt.alive
        assert "step4_runtimes" in report.completed_steps

    async def test_real_runtime_register_after_shutdown_fails(self):
        app = AgentApplication()
        await app.shutdown()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="int-2")
        with pytest.raises(CoordinatorClosedError):
            app.register_runtime(rt)

    async def test_context_manager_state(self):
        async with AgentApplication() as app:
            assert app.state is ApplicationState.RUNNING
        assert app.state is ApplicationState.STOPPED
