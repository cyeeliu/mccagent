"""Phase 8 生命周期守卫测试：P1-12 ~ P1-15。

验证 Coordinator 和 Application 的生命周期状态机、并发安全、
健康回调去重和 shutdown admission gate 已正确实现。

P1-12: Coordinator OPEN/CLOSING/CLOSED 状态机 + 注册事务。
P1-13: 健康回调去重（HEALTHY→UNHEALTHY 只触发一次故障回调）。
P1-14: Application STOPPING admission gate（shutdown 第一步关闭 Admission）。
P1-15: shutdown 超时和错误报告（ApplicationShutdownReport 完整字段）。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcc_core.runtime.capabilities import RuntimeCapabilities
from mcc_harness import RuntimeCoordinator, RuntimeStatus

# -- Runtime 桩 --


class _StubRuntime:
    """最小 Runtime 桩，支持 start/stop/cancel/health_check。"""

    def __init__(
        self,
        rid: str = "rt-1",
        *,
        healthy: bool = True,
        supports_health: bool = True,
    ) -> None:
        self._rid = rid
        self._healthy = healthy
        self._status = RuntimeStatus.RUNNING
        self._supports_health = supports_health
        self.start_called = False
        self.stop_called = False

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def agent_id(self) -> str:
        return self._rid

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status == RuntimeStatus.RUNNING

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_cancel=True,
            supports_health_check=self._supports_health,
        )

    async def start(self, *args: Any, **kwargs: Any) -> None:
        self.start_called = True
        self._status = RuntimeStatus.RUNNING

    async def stop(self, *, timeout: float = 5.0, **kwargs: Any) -> None:
        self.stop_called = True
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        self._status = RuntimeStatus.CANCELLED

    async def wait_done(self, *, timeout: float | None = None) -> Any:
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="done",
        )

    async def health_check(self) -> Any:
        from mcc_core.runtime.health import HealthState, RuntimeHealth

        state = HealthState.HEALTHY if self._healthy else HealthState.UNHEALTHY
        return RuntimeHealth(runtime_id=self._rid, state=state)


class _FlakyHealthRuntime(_StubRuntime):
    """健康状态可动态切换的 Runtime 桩，用于测试回调去重。"""

    def __init__(self, rid: str = "flaky-1") -> None:
        super().__init__(rid, healthy=True)
        self.health_sequence: list[bool] = []

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    async def health_check(self) -> Any:
        self.health_sequence.append(self._healthy)
        return await super().health_check()


# =========================================================================
# P1-12: Coordinator OPEN/CLOSING/CLOSED 状态机
# =========================================================================


class TestP1_12CoordinatorStateMachine:
    """P1-12: Coordinator 生命周期状态 OPEN → CLOSING → CLOSED。"""

    def test_initial_state_is_open(self):
        """新构造的 Coordinator 状态应为 OPEN。"""
        from mcc_harness.runtime.coordinator import CoordinatorState

        coord = RuntimeCoordinator()
        assert coord.state == CoordinatorState.OPEN
        assert not coord.is_closing
        assert not coord.is_closed

    @pytest.mark.asyncio
    async def test_begin_shutdown_transitions_to_closing(self):
        """begin_shutdown 应将状态从 OPEN 切换到 CLOSING。"""
        from mcc_harness.runtime.coordinator import CoordinatorState

        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        assert coord.state == CoordinatorState.CLOSING
        assert coord.is_closing
        assert not coord.is_closed

    @pytest.mark.asyncio
    async def test_begin_shutdown_is_idempotent(self):
        """begin_shutdown 幂等：多次调用不改变状态。"""
        from mcc_harness.runtime.coordinator import CoordinatorState

        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        await coord.begin_shutdown()
        assert coord.state == CoordinatorState.CLOSING

    @pytest.mark.asyncio
    async def test_register_rejected_after_begin_shutdown(self):
        """begin_shutdown 后 register 应抛 CoordinatorClosedError。"""
        from mcc_harness.runtime.coordinator import CoordinatorClosedError

        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        rt = _StubRuntime("rt-reject")
        with pytest.raises(CoordinatorClosedError):
            coord.register(rt)

    @pytest.mark.asyncio
    async def test_register_and_start_rejected_after_begin_shutdown(self):
        """begin_shutdown 后 register_and_start 应抛 CoordinatorClosedError。"""
        from mcc_harness.runtime.coordinator import CoordinatorClosedError

        coord = RuntimeCoordinator()
        await coord.begin_shutdown()
        rt = _StubRuntime("rt-reject-2")
        with pytest.raises(CoordinatorClosedError):
            await coord.register_and_start(rt)

    @pytest.mark.asyncio
    async def test_shutdown_all_transitions_to_closed(self):
        """shutdown_all 完成后状态应为 CLOSED。"""
        from mcc_harness.runtime.coordinator import CoordinatorState

        coord = RuntimeCoordinator()
        await coord.shutdown_all(timeout=1.0)
        assert coord.state == CoordinatorState.CLOSED
        assert coord.is_closing
        assert coord.is_closed

    @pytest.mark.asyncio
    async def test_shutdown_all_idempotent_after_closed(self):
        """CLOSED 后再次 shutdown_all 返回空报告。"""
        coord = RuntimeCoordinator()
        await coord.shutdown_all(timeout=1.0)
        report = await coord.shutdown_all(timeout=1.0)
        assert not report.has_errors
        assert report.total == 0


# =========================================================================
# P1-13: 健康回调去重
# =========================================================================


class TestP1_13HealthCallbackDedup:
    """P1-13: HEALTHY→UNHEALTHY 只触发一次故障回调，恢复后重置。"""

    @pytest.mark.asyncio
    async def test_monitor_state_tracks_failure_reported(self):
        """RuntimeMonitorState 应有 failure_reported 字段用于去重。"""
        from mcc_harness.runtime.coordinator import RuntimeMonitorState

        state = RuntimeMonitorState()
        assert state.failure_reported is False
        assert state.last_health == "healthy"

    @pytest.mark.asyncio
    async def test_unhealthy_callback_fired_once(self):
        """连续两轮 UNHEALTHY 只触发一次故障回调。"""
        coord = RuntimeCoordinator()
        rt = _FlakyHealthRuntime("flaky-dedup")
        await coord.register_and_start(rt)

        callback_count = 0

        async def on_exit(rid: str, status: Any) -> None:
            nonlocal callback_count
            callback_count += 1

        coord.on_exit(on_exit)
        await coord.start_health_monitor(interval=0.05)

        # 第一轮：unhealthy → 触发回调
        rt.set_healthy(False)
        await asyncio.sleep(0.15)

        # 第二轮：仍 unhealthy → 不应再次触发
        await asyncio.sleep(0.15)

        await coord.stop_health_monitor()
        await coord.shutdown_all(timeout=1.0)

        assert callback_count == 1, (
            f"expected exactly 1 callback for persistent unhealthy, got {callback_count}"
        )

    @pytest.mark.asyncio
    async def test_recovery_resets_failure_flag(self):
        """UNHEALTHY→HEALTHY 恢复后重置 failure_reported，可再次触发。"""
        coord = RuntimeCoordinator()
        rt = _FlakyHealthRuntime("flaky-recover")
        await coord.register_and_start(rt)

        callback_count = 0

        async def on_exit(rid: str, status: Any) -> None:
            nonlocal callback_count
            callback_count += 1

        coord.on_exit(on_exit)
        await coord.start_health_monitor(interval=0.05)

        # 第一轮：unhealthy → 触发回调
        rt.set_healthy(False)
        await asyncio.sleep(0.15)
        assert callback_count >= 1

        # 恢复
        rt.set_healthy(True)
        await asyncio.sleep(0.15)

        # 再次 unhealthy → 应再次触发
        rt.set_healthy(False)
        await asyncio.sleep(0.15)

        await coord.stop_health_monitor()
        await coord.shutdown_all(timeout=1.0)

        assert callback_count >= 2, (
            f"expected >=2 callbacks after recovery+re-failure, got {callback_count}"
        )


# =========================================================================
# P1-14: Application STOPPING admission gate
# =========================================================================


class TestP1_14ApplicationAdmissionGate:
    """P1-14: Application shutdown 第一步关闭 Admission。"""

    def test_application_state_enum_exists(self):
        """ApplicationState 应有 CREATED/RUNNING/STOPPING/STOPPED。"""
        from mcc_cli.application import ApplicationState

        assert ApplicationState.CREATED < ApplicationState.RUNNING
        assert ApplicationState.RUNNING < ApplicationState.STOPPING
        assert ApplicationState.STOPPING < ApplicationState.STOPPED

    def test_initial_state_is_running(self):
        """新构造的 Application 状态应为 RUNNING。"""
        from mcc_cli.application import AgentApplication, ApplicationState

        app = AgentApplication()
        assert app.state == ApplicationState.RUNNING
        assert not app.is_stopping

    @pytest.mark.asyncio
    async def test_shutdown_transitions_to_stopping_then_stopped(self):
        """shutdown 应先切换到 STOPPING，完成后切换到 STOPPED。"""
        from mcc_cli.application import AgentApplication, ApplicationState

        app = AgentApplication()
        await app.shutdown(timeout=2.0)
        assert app.state == ApplicationState.STOPPED
        assert app.is_stopping

    @pytest.mark.asyncio
    async def test_shutdown_calls_coordinator_begin_shutdown(self):
        """shutdown 第一步应调用 coordinator.begin_shutdown（关闭 Admission）。"""
        from mcc_cli.application import AgentApplication
        from mcc_harness.runtime.coordinator import CoordinatorState

        coord = RuntimeCoordinator()
        app = AgentApplication(coordinator=coord)
        await app.shutdown(timeout=2.0)

        # Coordinator 应已进入 CLOSED（经过 CLOSING）
        assert coord.state == CoordinatorState.CLOSED

    @pytest.mark.asyncio
    async def test_register_rejected_after_shutdown(self):
        """shutdown 后 register 应抛 CoordinatorClosedError。"""
        from mcc_cli.application import AgentApplication
        from mcc_harness.runtime.coordinator import CoordinatorClosedError

        coord = RuntimeCoordinator()
        app = AgentApplication(coordinator=coord)
        await app.shutdown(timeout=2.0)

        rt = _StubRuntime("rt-after-shutdown")
        with pytest.raises(CoordinatorClosedError):
            coord.register(rt)


# =========================================================================
# P1-15: shutdown 超时和错误报告
# =========================================================================


class TestP1_15ShutdownReport:
    """P1-15: ApplicationShutdownReport 完整字段 + 并发 shutdown。"""

    def test_shutdown_report_has_all_fields(self):
        """ApplicationShutdownReport 应有完整字段。"""
        from mcc_cli.application import ApplicationShutdownReport

        report = ApplicationShutdownReport()
        assert report.completed_steps == []
        assert report.failed_steps == {}
        assert report.timed_out_steps == []
        assert report.runtime_report is None
        assert report.elapsed_seconds == 0.0
        assert not report.has_errors

    @pytest.mark.asyncio
    async def test_shutdown_report_populated_after_shutdown(self):
        """shutdown 后应返回包含 completed_steps 的报告。"""
        from mcc_cli.application import AgentApplication

        app = AgentApplication()
        report = await app.shutdown(timeout=5.0)
        assert "step1_admission" in report.completed_steps
        assert "step3_cancel_all" in report.completed_steps
        assert "step4_runtimes" in report.completed_steps
        assert "step7_clear_terminal" in report.completed_steps
        assert report.elapsed_seconds >= 0.0
        assert app.shutdown_report is report or app.shutdown_report is not None

    @pytest.mark.asyncio
    async def test_concurrent_shutdown_shares_same_task(self):
        """并发 shutdown 调用应等待同一个 task，不重复执行。"""
        from mcc_cli.application import AgentApplication

        app = AgentApplication()
        # 同时发起两个 shutdown
        report1, report2 = await asyncio.gather(
            app.shutdown(timeout=5.0),
            app.shutdown(timeout=5.0),
        )
        # 两者应返回相同的报告（同一个 task 的结果）
        assert report1.completed_steps == report2.completed_steps

    @pytest.mark.asyncio
    async def test_shutdown_report_with_mcp_failure(self):
        """MCP 关闭失败应记录在 failed_steps 中。"""
        from mcc_cli.application import AgentApplication

        class FailingMcp:
            async def close_all(self) -> None:
                raise RuntimeError("mcp close failed")

        app = AgentApplication(mcp_manager=FailingMcp())
        report = await app.shutdown(timeout=5.0)
        assert "step5_mcp" in report.failed_steps
        assert "mcp close failed" in report.failed_steps["step5_mcp"]
