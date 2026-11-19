"""Phase 5: RuntimeCoordinator 生命周期闭环测试。

验收标准：
- Coordinator 是唯一 Runtime 生命周期登记表。
- 注册失败无半注册对象（register_and_start 事务回滚）。
- Runtime 崩溃最终被标记和注销（健康监控）。
- 并发 register/unregister 不产生悬挂任务。
- shutdown_all 支持总超时和优先级排序。
- 自然退出回调在 Runtime 到达终态时触发。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from mcc_core.runtime import RuntimeStatus as V2RuntimeStatus
from mcc_core.runtime.capabilities import RuntimeCapabilities
from mcc_harness import (
    InteractiveRuntimeV2,
    RuntimeCoordinator,
    RuntimeStatus,
    ShutdownReport,
)


class FakeAgent:
    """最小 Agent 桩，满足 InteractiveRuntimeV2 构造。"""

    async def run(self, messages, *, tools=None, system=None, model=None):
        yield


class FailingStartRuntime:
    """start() 抛异常的 Runtime 桩，用于测试事务回滚。"""

    def __init__(self) -> None:
        self._rid = "fail-1"

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def status(self) -> RuntimeStatus:
        return RuntimeStatus.CREATED

    @property
    def alive(self) -> bool:
        return False

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    async def start(self, task: str = "", *, task_id=None) -> None:
        raise RuntimeError("start failed")

    async def stop(self, *, timeout: float = 5.0) -> None:
        pass

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=RuntimeStatus.FAILED,
            summary="start failed",
        )


class SlowStopRuntime:
    """stop() 超时的 Runtime 桩，用于测试总超时。"""

    def __init__(self, rid: str = "slow-1") -> None:
        self._rid = rid
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

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    async def start(self, task: str = "", *, task_id=None) -> None:
        pass

    async def stop(self, *, timeout: float = 5.0) -> None:
        await asyncio.sleep(100)
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="slow",
        )


class HealthCheckRuntime:
    """支持 health_check 的 Runtime 桩。"""

    def __init__(self, rid: str = "hc-1", healthy: bool = True) -> None:
        self._rid = rid
        self._healthy = healthy
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

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=True)

    async def start(self, task: str = "", *, task_id=None) -> None:
        pass

    async def stop(self, *, timeout: float = 5.0) -> None:
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="done",
        )

    async def health_check(self):
        from mcc_core.runtime.health import HealthState, RuntimeHealth

        state = HealthState.HEALTHY if self._healthy else HealthState.UNHEALTHY
        return RuntimeHealth(runtime_id=self._rid, state=state)


# -- register_and_start 事务回滚 --


class TestRegisterAndStart:
    async def test_register_and_start_success(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ras-1")
        await coord.register_and_start(rt)
        assert coord.get("ras-1") is rt
        assert rt.status is V2RuntimeStatus.READY
        await coord.shutdown_all()

    async def test_register_and_start_rollback_on_failure(self):
        coord = RuntimeCoordinator()
        rt = FailingStartRuntime()
        with pytest.raises(RuntimeError, match="start failed"):
            await coord.register_and_start(rt)
        assert coord.get("fail-1") is None
        assert coord.count == 0

    async def test_register_and_start_with_priority(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ras-p1")
        await coord.register_and_start(rt, priority=5)
        assert coord.get("ras-p1") is rt
        await coord.shutdown_all()


# -- 优先级关闭排序 --


class TestPriorityShutdown:
    async def test_shutdown_priority_order(self):
        """priority 数值越小越先停止。"""
        coord = RuntimeCoordinator()
        stop_order: list[str] = []

        class TrackingRuntime:
            def __init__(self, rid: str) -> None:
                self._rid = rid
                self._status = RuntimeStatus.RUNNING

            @property
            def runtime_id(self) -> str:
                return self._rid

            @property
            def status(self) -> RuntimeStatus:
                return self._status

            @property
            def alive(self) -> bool:
                return self._status is RuntimeStatus.RUNNING

            @property
            def capabilities(self) -> RuntimeCapabilities:
                return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

            async def start(self, task: str = "", *, task_id=None) -> None:
                pass

            async def stop(self, *, timeout: float = 5.0) -> None:
                stop_order.append(self._rid)
                self._status = RuntimeStatus.SHUTDOWN

            def cancel(self) -> None:
                pass

            async def wait_done(self, *, timeout=None):
                from mcc_harness import RuntimeResult

                return RuntimeResult(
                    runtime_id=self._rid,
                    status=self._status,
                    summary="done",
                )

        rt_low = TrackingRuntime("low")
        rt_high = TrackingRuntime("high")
        rt_mid = TrackingRuntime("mid")
        coord.register(rt_low, priority=1)
        coord.register(rt_high, priority=10)
        coord.register(rt_mid, priority=5)

        await coord.shutdown_all()
        assert stop_order == ["low", "mid", "high"]

    async def test_default_priority_zero(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="dp-1")
        coord.register(rt)
        assert coord.get("dp-1") is rt
        await coord.shutdown_all()


# -- ShutdownReport 增强 --


class TestShutdownReportEnhanced:
    def test_timed_out_field(self):
        report = ShutdownReport()
        report.timed_out.append("to-1")
        assert report.has_errors
        assert report.total == 1

    def test_stopped_and_failed_and_timed_out(self):
        report = ShutdownReport()
        report.stopped.append("ok-1")
        report.failed["err-1"] = "boom"
        report.timed_out.append("to-1")
        assert report.total == 3
        assert report.has_errors

    def test_empty_report_no_errors(self):
        report = ShutdownReport()
        assert not report.has_errors
        assert report.total == 0


# -- shutdown_all 总超时 --


class TestShutdownAllTimeout:
    async def test_total_timeout_produces_timed_out(self):
        coord = RuntimeCoordinator()
        rt = SlowStopRuntime("to-1")
        coord.register(rt)
        report = await coord.shutdown_all(timeout=0.1)
        assert "to-1" in report.timed_out or "to-1" in report.failed
        assert report.has_errors

    async def test_shutdown_all_alias_stop_all(self):
        """stop_all 是 shutdown_all 的向后兼容别名。"""
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="alias-1")
        coord.register(rt)
        await rt.start()
        report = await coord.stop_all()
        assert "alias-1" in report.stopped


# -- is_closed 属性 --


class TestIsClosed:
    async def test_is_closed_false_initially(self):
        coord = RuntimeCoordinator()
        assert not coord.is_closed

    async def test_is_closed_true_after_shutdown(self):
        coord = RuntimeCoordinator()
        await coord.shutdown_all()
        assert coord.is_closed


# -- 幂等 unregister --


class TestIdempotentUnregister:
    def test_unregister_twice_returns_none(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="iu-1")
        coord.register(rt)
        first = coord.unregister("iu-1")
        second = coord.unregister("iu-1")
        assert first is rt
        assert second is None

    def test_unregister_cleans_priority(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="up-1")
        coord.register(rt, priority=5)
        coord.unregister("up-1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="up-1")
        coord.register(rt2, priority=3)
        assert coord.get("up-1") is rt2


# -- 健康监控 --


class TestHealthMonitor:
    async def test_start_stop_health_monitor(self):
        coord = RuntimeCoordinator()
        await coord.start_health_monitor(interval=0.01)
        assert coord._health_task is not None
        await coord.stop_health_monitor()
        assert coord._health_task is None

    async def test_health_monitor_detects_unhealthy(self):
        coord = RuntimeCoordinator()
        rt = HealthCheckRuntime("hm-1", healthy=False)
        coord.register(rt)

        unhealthy_detected: list[str] = []

        async def on_unhealthy(rid: str, status: RuntimeStatus) -> None:
            unhealthy_detected.append(rid)

        coord.on_exit(on_unhealthy)
        await coord.start_health_monitor(interval=0.05)
        await asyncio.sleep(0.2)
        await coord.stop_health_monitor()
        assert "hm-1" in unhealthy_detected
        await coord.shutdown_all()

    async def test_health_monitor_skips_no_health_check(self):
        """旧 Protocol Runtime 没有 health_check 方法，监控应跳过。"""
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="old-1")
        coord.register(rt)
        await coord.start_health_monitor(interval=0.01)
        await asyncio.sleep(0.05)
        await coord.stop_health_monitor()
        await coord.shutdown_all()


# -- 自然退出回调 --


class TestExitCallback:
    async def test_exit_callback_on_terminal(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ec-1")
        coord.register(rt)
        await rt.start()

        exited: list[tuple[str, RuntimeStatus]] = []

        async def on_exit(rid: str, status: RuntimeStatus) -> None:
            exited.append((rid, status))

        coord.on_exit(on_exit)
        await coord.start_exit_monitor()
        await rt.stop()
        await asyncio.sleep(1.0)
        await coord.stop_exit_monitor()
        assert any(rid == "ec-1" for rid, _ in exited)

    async def test_exit_monitor_idempotent_start(self):
        coord = RuntimeCoordinator()
        await coord.start_exit_monitor()
        task1 = coord._exit_monitor_task
        await coord.start_exit_monitor()
        task2 = coord._exit_monitor_task
        assert task1 is task2
        await coord.stop_exit_monitor()

    async def test_exit_callback_exception_does_not_crash_monitor(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="exc-1")
        coord.register(rt)
        await rt.start()

        async def bad_callback(rid: str, status: RuntimeStatus) -> None:
            raise ValueError("callback error")

        async def good_callback(rid: str, status: RuntimeStatus) -> None:
            pass

        coord.on_exit(bad_callback)
        coord.on_exit(good_callback)
        await coord.start_exit_monitor()
        await rt.stop()
        await asyncio.sleep(1.0)
        await coord.stop_exit_monitor()
        assert coord._exit_monitor_task is None


# -- 并发安全 --


class TestConcurrentSafety:
    async def test_concurrent_register_different_ids(self):
        coord = RuntimeCoordinator()
        n = 20

        async def register_one(i: int) -> None:
            coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id=f"cc-{i}"))

        await asyncio.gather(*(register_one(i) for i in range(n)))
        assert coord.count == n
        await coord.shutdown_all()

    async def test_concurrent_register_duplicate_raises(self):
        coord = RuntimeCoordinator()
        errors: list[Exception] = []

        async def try_register() -> None:
            try:
                coord.register(InteractiveRuntimeV2(FakeAgent(), runtime_id="dup-cc"))
            except RuntimeError as exc:
                errors.append(exc)

        await asyncio.gather(*(try_register() for _ in range(10)))
        assert len(errors) == 9
        assert coord.count == 1
        await coord.shutdown_all()


# -- get_by_agent / health_check_all --


class TestGetByAgent:
    """RuntimeCoordinator.get_by_agent 行为。"""

    def test_find_by_agent_id(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r1",
            agent_id="agent-alpha",
        )
        coord.register(rt)
        found = coord.get_by_agent("agent-alpha")
        assert found is rt

    def test_return_none_when_not_found(self):
        coord = RuntimeCoordinator()
        assert coord.get_by_agent("no-such-agent") is None

    def test_find_among_multiple(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r1",
            agent_id="a1",
        )
        rt2 = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r2",
            agent_id="a2",
        )
        rt3 = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r3",
            agent_id="a3",
        )
        coord.register(rt1)
        coord.register(rt2)
        coord.register(rt3)
        assert coord.get_by_agent("a2") is rt2

    def test_returns_first_match(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r1",
            agent_id="dup-agent",
        )
        rt2 = InteractiveRuntimeV2(
            FakeAgent(),
            runtime_id="r2",
            agent_id="dup-agent",
        )
        coord.register(rt1)
        coord.register(rt2)
        found = coord.get_by_agent("dup-agent")
        assert found in (rt1, rt2)


class _HealthCheckStub:
    """带 health_check 的 Runtime 桩。"""

    def __init__(self, rid: str, healthy: bool = True) -> None:
        self._rid = rid
        self._healthy = healthy

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def agent_id(self) -> str:
        return self._rid

    @property
    def status(self) -> RuntimeStatus:
        return RuntimeStatus.IDLE

    @property
    def alive(self) -> bool:
        return True

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=True)

    async def start(self, task: str = "", *, task_id=None) -> None:
        pass

    async def stop(self, *, timeout: float = 5.0) -> None:
        pass

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeStatus.COMPLETED, RuntimeResult.success(self._rid)

    async def health_check(self) -> Any:
        return {"healthy": self._healthy}


class TestHealthCheckAll:
    """RuntimeCoordinator.health_check_all 行为。"""

    async def test_empty_coordinator_returns_empty_dict(self):
        coord = RuntimeCoordinator()
        result = await coord.health_check_all()
        assert result == {}

    async def test_skips_runtime_without_health_check(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1")
        coord.register(rt)
        result = await coord.health_check_all()
        assert "r1" in result

    async def test_collects_health_from_supporting_runtimes(self):
        coord = RuntimeCoordinator()
        rt1 = _HealthCheckStub("hc-1", healthy=True)
        rt2 = _HealthCheckStub("hc-2", healthy=True)
        coord.register(rt1)
        coord.register(rt2)
        result = await coord.health_check_all()
        assert set(result.keys()) == {"hc-1", "hc-2"}

    async def test_health_check_exception_is_logged_not_raised(self):
        coord = RuntimeCoordinator()

        class _FailingHealth:
            def __init__(self) -> None:
                self._rid = "fail-hc"

            @property
            def runtime_id(self) -> str:
                return self._rid

            @property
            def agent_id(self) -> str:
                return self._rid

            @property
            def status(self) -> RuntimeStatus:
                return RuntimeStatus.IDLE

            @property
            def alive(self) -> bool:
                return True

            @property
            def capabilities(self) -> RuntimeCapabilities:
                return RuntimeCapabilities(supports_cancel=True, supports_health_check=True)

            async def start(self, task: str = "", *, task_id=None) -> None:
                pass

            async def stop(self, *, timeout: float = 5.0) -> None:
                pass

            def cancel(self) -> None:
                pass

            async def wait_done(self, *, timeout=None):
                from mcc_harness import RuntimeResult

                return RuntimeStatus.COMPLETED, RuntimeResult.success(self._rid)

            async def health_check(self):
                raise RuntimeError("health check failed")

        coord.register(_FailingHealth())
        result = await coord.health_check_all()
        assert "fail-hc" not in result
