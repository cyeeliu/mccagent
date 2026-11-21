"""PR-13: Coordinator stop timeout recovery（P1-05 + P1-12）。

验证：
- P1-05: shutdown_all 总超时后保存 stop Task，cleanup grace period 等待，
  仍未退出的 Runtime 记入 still_running。
- P1-05: ShutdownReport 包含 still_running / cleanup_pending 字段。
- P1-12: register_async / unregister_async 使用锁的 async 版本。
- P1-12: 同步 register 文档明确单事件循环线程契约。
"""

from __future__ import annotations

import asyncio
import contextlib

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
    async def run(self, task: str = ""):
        return type("R", (), {"final_text": "ok"})()

    async def run_streaming(self, task: str = "", queue=None, **kw):
        yield type("R", (), {"final_text": "ok"})()


class CancelAwareSlowStop:
    """stop() 阻塞但 cancel() 可使其快速退出的 Runtime 桩。"""

    def __init__(self, rid: str = "cas-1") -> None:
        self._rid = rid
        self._status = RuntimeStatus.RUNNING
        self._cancelled = asyncio.Event()

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status not in (
            RuntimeStatus.SHUTDOWN,
            RuntimeStatus.FAILED,
        )

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    async def start(self, task: str = "", *, task_id=None) -> None:
        pass

    async def stop(self, *, timeout: float = 5.0) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._cancelled.wait(), timeout=timeout)
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        self._cancelled.set()

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="done",
        )


class HangingStop:
    """stop() 永不退出的 Runtime 桩，cancel 也不响应。"""

    def __init__(self, rid: str = "hang-1") -> None:
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
        await asyncio.sleep(10000)

    def cancel(self) -> None:
        pass

    async def wait_done(self, *, timeout=None):
        from mcc_harness import RuntimeResult

        return RuntimeResult(
            runtime_id=self._rid,
            status=self._status,
            summary="hang",
        )


class SlowButFiniteStop:
    """stop() 需要短时间完成，总超时后 cleanup grace 内可完成。"""

    def __init__(self, rid: str = "sbf-1", stop_delay: float = 0.3) -> None:
        self._rid = rid
        self._stop_delay = stop_delay
        self._status = RuntimeStatus.RUNNING

    @property
    def runtime_id(self) -> str:
        return self._rid

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status == RuntimeStatus.RUNNING

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    async def start(self, task: str = "", *, task_id=None) -> None:
        pass

    async def stop(self, *, timeout: float = 5.0) -> None:
        await asyncio.sleep(self._stop_delay)
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


# -- P1-05: ShutdownReport 新字段 --


class TestShutdownReportFields:
    def test_still_running_field_exists(self):
        report = ShutdownReport()
        assert hasattr(report, "still_running")
        assert report.still_running == []

    def test_cleanup_pending_field_exists(self):
        report = ShutdownReport()
        assert hasattr(report, "cleanup_pending")
        assert report.cleanup_pending == []

    def test_has_errors_with_still_running(self):
        report = ShutdownReport()
        report.still_running.append("rt-1")
        assert report.has_errors

    def test_total_counts_still_running(self):
        report = ShutdownReport(stopped=["a"], still_running=["b", "c"])
        assert report.total == 3


# -- P1-05: shutdown_all 总超时后 cleanup grace 等待 --


class TestShutdownAllCleanupGrace:
    async def test_cleanup_grace_recovers_slow_stop(self):
        """总超时后 cleanup grace 内完成的 Runtime 计入 stopped。"""
        coord = RuntimeCoordinator()
        rt = SlowButFiniteStop("grace-1", stop_delay=0.3)
        coord.register(rt)
        report = await coord.shutdown_all(
            timeout=0.1,
            cleanup_grace=1.0,
        )
        assert "grace-1" in report.stopped
        assert report.still_running == []

    async def test_still_running_for_hanging_stop(self):
        """总超时且 cleanup grace 后仍未退出的 Runtime 计入 still_running。"""
        coord = RuntimeCoordinator()
        rt = HangingStop("hang-1")
        coord.register(rt)
        report = await coord.shutdown_all(
            timeout=0.1,
            cleanup_grace=0.2,
        )
        assert "hang-1" in report.still_running

    async def test_cancel_then_cleanup_stops_runtime(self):
        """总超时后发 cancel，cleanup grace 内 Runtime 停止。"""
        coord = RuntimeCoordinator()
        rt = CancelAwareSlowStop("cas-1")
        coord.register(rt)
        report = await coord.shutdown_all(
            timeout=0.1,
            cleanup_grace=1.0,
        )
        assert "cas-1" in report.stopped
        assert report.still_running == []

    async def test_mixed_stop_outcomes(self):
        """混合场景：一个快速停止、一个 cleanup grace 内停止、一个挂起。"""
        coord = RuntimeCoordinator()
        rt_fast = SlowButFiniteStop("fast-1", stop_delay=0.01)
        rt_grace = SlowButFiniteStop("grace-1", stop_delay=0.3)
        rt_hang = HangingStop("hang-1")
        coord.register(rt_fast, priority=1)
        coord.register(rt_grace, priority=2)
        coord.register(rt_hang, priority=3)
        report = await coord.shutdown_all(
            timeout=0.1,
            cleanup_grace=0.5,
        )
        assert "fast-1" in report.stopped
        assert "grace-1" in report.stopped
        assert "hang-1" in report.still_running

    async def test_no_leaked_tasks_after_shutdown(self):
        """shutdown_all 完成后不应遗留未完成的 stop Task。"""
        coord = RuntimeCoordinator()
        rt = HangingStop("hang-1")
        coord.register(rt)
        await coord.shutdown_all(timeout=0.05, cleanup_grace=0.1)
        pending = [
            t
            for t in asyncio.all_tasks()
            if not t.done() and t is not asyncio.current_task()
        ]
        stop_related = [
            t
            for t in pending
            if "stop" in t.get_name() or "safe_stop" in t.get_name()
        ]
        assert stop_related == []


# -- P1-12: async register/unregister --


class TestAsyncRegister:
    async def test_register_async_basic(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ar-1")
        await coord.register_async(rt)
        assert coord.get("ar-1") is rt
        await coord.shutdown_all()

    async def test_register_async_with_priority(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="ar-p1")
        await coord.register_async(rt, priority=10)
        assert coord._priorities["ar-p1"] == 10
        await coord.shutdown_all()

    async def test_register_async_duplicate_raises(self):
        coord = RuntimeCoordinator()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="dup-1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="dup-1")
        await coord.register_async(rt1)
        with pytest.raises(RuntimeError, match="duplicate"):
            await coord.register_async(rt2)
        await coord.shutdown_all()

    async def test_register_async_closed_raises(self):
        coord = RuntimeCoordinator()
        await coord.shutdown_all()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="closed-1")
        with pytest.raises(Exception, match="closed"):
            await coord.register_async(rt)

    async def test_unregister_async_basic(self):
        coord = RuntimeCoordinator()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="un-1")
        await coord.register_async(rt)
        removed = await coord.unregister_async("un-1")
        assert removed is rt
        assert coord.get("un-1") is None

    async def test_unregister_async_idempotent(self):
        coord = RuntimeCoordinator()
        result = await coord.unregister_async("nonexistent")
        assert result is None

    async def test_concurrent_register_async_safe(self):
        """并发 register_async 不同 ID 不产生数据竞争。"""
        coord = RuntimeCoordinator()
        runtimes = [
            InteractiveRuntimeV2(FakeAgent(), runtime_id=f"conc-{i}")
            for i in range(10)
        ]
        await asyncio.gather(*(coord.register_async(rt) for rt in runtimes))
        assert coord.count == 10
        await coord.shutdown_all()
