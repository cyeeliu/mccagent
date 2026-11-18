"""RuntimeCoordinator：运行时生命周期协调器。

统一管理 Runtime 的注册、注销、查询、停止和关闭清理。
提供活跃 Runtime 列表和状态快照，聚合关闭异常。
支持注册事务回滚、健康监控、自然退出回调和优先级关闭。
不执行规划任务或选择 Agent，仅负责生命周期管理。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .base import _TERMINAL_STATUSES, RuntimeStatus

logger = logging.getLogger(__name__)


class CoordinatorClosedError(RuntimeError):
    """Coordinator 已关闭，不再接受注册。"""


@dataclass
class RuntimeSnapshot:
    """运行时状态快照。"""

    runtime_id: str
    status: RuntimeStatus
    alive: bool


@dataclass
class ShutdownReport:
    """关闭报告：汇总 shutdown_all 的结果和异常。"""

    stopped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.failed) or bool(self.timed_out)

    @property
    def total(self) -> int:
        return len(self.stopped) + len(self.failed) + len(self.timed_out)


ExitCallback = Callable[[str, RuntimeStatus], Awaitable[None]]


class RuntimeCoordinator:
    """运行时生命周期协调器。

    职责：
    - Runtime 注册、注销和查询（重复 ID 抛 RuntimeError）。
    - 活跃 Runtime 列表和状态快照。
    - shutdown_all 统一关闭，幂等，聚合异常，支持总超时和优先级排序。
    - 终态 Runtime 归档（可选 TTL 清理）。
    - 注册事务：register_and_start 在启动失败时自动回滚注销。
    - 健康监控：后台任务定期检查 Runtime 健康，崩溃自动标记和注销。
    - 自然退出回调：Runtime 到达终态时通知已注册的回调。

    不负责规划任务或选择 Agent。
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, Any] = {}
        self._priorities: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._stopped = False
        self._health_task: asyncio.Task[None] | None = None
        self._health_interval: float = 30.0
        self._exit_callbacks: list[ExitCallback] = []
        self._exit_monitor_task: asyncio.Task[None] | None = None
        self._exit_monitor_runtimes: set[str] = set()

    def register(
        self,
        runtime: Any,
        *,
        priority: int = 0,
    ) -> None:
        """注册 Runtime，重复 ID 抛 RuntimeError，已关闭抛 CoordinatorClosedError。

        priority 控制关闭顺序：数值越小越先停止（依赖方优先）。
        """
        if self._stopped:
            raise CoordinatorClosedError(
                "coordinator is closed; create a new one to register runtimes"
            )
        rid = runtime.runtime_id
        if rid in self._runtimes:
            raise RuntimeError(f"duplicate runtime id: {rid}")
        self._runtimes[rid] = runtime
        self._priorities[rid] = priority

    async def register_and_start(
        self,
        runtime: Any,
        *,
        priority: int = 0,
    ) -> None:
        """注册并启动 Runtime，启动失败时自动回滚注销。

        事务语义：要么 Runtime 注册且启动成功，要么不留任何半注册对象。
        """
        self.register(runtime, priority=priority)
        try:
            start_coro = runtime.start()
            if asyncio.iscoroutine(start_coro):
                await start_coro
        except Exception:
            self.unregister(runtime.runtime_id)
            raise

    def unregister(self, runtime_id: str) -> Any | None:
        """注销 Runtime，返回被移除的实例（不存在则返回 None，幂等）。"""
        self._priorities.pop(runtime_id, None)
        self._exit_monitor_runtimes.discard(runtime_id)
        return self._runtimes.pop(runtime_id, None)

    def get(self, runtime_id: str) -> Any | None:
        """按 ID 查询 Runtime（不存在返回 None）。"""
        return self._runtimes.get(runtime_id)

    def get_by_agent(self, agent_id: str) -> Any | None:
        """按 agent_id 查询 Runtime（不存在返回 None）。"""
        for rt in self._runtimes.values():
            if rt.agent_id == agent_id:
                return rt
        return None

    async def health_check_all(self) -> dict[str, Any]:
        """对所有已注册 Runtime 执行 health_check，返回 {runtime_id: health}。

        不支持 health_check 的 Runtime 跳过。
        """
        results: dict[str, Any] = {}
        for rid, rt in list(self._runtimes.items()):
            if not rt.capabilities.supports_health_check:
                continue
            try:
                results[rid] = await rt.health_check()
            except Exception:
                logger.exception("health_check failed for runtime %s", rid)
        return results

    def list_active(self) -> list[Any]:
        """返回所有活跃（alive）的 Runtime。"""
        return [rt for rt in self._runtimes.values() if rt.alive]

    def list_all(self) -> list[Any]:
        """返回所有已注册的 Runtime。"""
        return list(self._runtimes.values())

    def snapshot(self) -> list[RuntimeSnapshot]:
        """返回所有 Runtime 的状态快照。"""
        return [
            RuntimeSnapshot(
                runtime_id=rt.runtime_id,
                status=rt.status,
                alive=rt.alive,
            )
            for rt in self._runtimes.values()
        ]

    def on_exit(self, callback: ExitCallback) -> None:
        """注册自然退出回调。

        回调签名: async def callback(runtime_id: str, status: RuntimeStatus) -> None。
        当监控检测到 Runtime 到达终态时调用。
        需先调用 start_exit_monitor 激活监控。
        """
        self._exit_callbacks.append(callback)

    async def start_exit_monitor(self) -> None:
        """启动自然退出监控，检测 Runtime 到达终态并触发回调。"""
        if self._exit_monitor_task is not None:
            return
        self._exit_monitor_task = asyncio.create_task(self._exit_monitor_loop())

    async def stop_exit_monitor(self) -> None:
        """停止自然退出监控。"""
        if self._exit_monitor_task is not None:
            self._exit_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._exit_monitor_task
            self._exit_monitor_task = None

    async def _exit_monitor_loop(self) -> None:
        """自然退出监控循环：轮询 Runtime 状态，到达终态时触发回调。"""
        while not self._stopped:
            await asyncio.sleep(0.5)
            for rid, rt in list(self._runtimes.items()):
                if rid in self._exit_monitor_runtimes:
                    continue
                if not rt.alive and rt.status in _TERMINAL_STATUSES:
                    self._exit_monitor_runtimes.add(rid)
                    for cb in self._exit_callbacks:
                        try:
                            await cb(rid, rt.status)
                        except Exception:
                            logger.exception(
                                "exit callback failed for runtime %s",
                                rid,
                            )

    async def start_health_monitor(
        self,
        interval: float = 30.0,
    ) -> None:
        """启动健康监控后台任务，定期检查各 Runtime 健康状态。

        检测到 UNHEALTHY 的 Runtime 会记录日志并触发退出回调。
        """
        if self._health_task is not None:
            return
        self._health_interval = interval
        self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def stop_health_monitor(self) -> None:
        """停止健康监控后台任务。"""
        if self._health_task is not None:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None

    async def _health_monitor_loop(self) -> None:
        """健康监控循环：定期调用各 Runtime 的 health_check。"""
        while not self._stopped:
            await asyncio.sleep(self._health_interval)
            for rid, rt in list(self._runtimes.items()):
                if not rt.capabilities.supports_health_check:
                    continue
                try:
                    health = await rt.health_check()
                    if (
                        hasattr(health, "state")
                        and str(health.state) == "unhealthy"
                    ):
                        logger.warning(
                            "runtime %s is unhealthy: %s",
                            rid,
                            health,
                        )
                        for cb in self._exit_callbacks:
                            try:
                                await cb(rid, rt.status)
                            except Exception:
                                logger.exception(
                                    "exit callback failed for runtime %s",
                                    rid,
                                )
                except Exception:
                    logger.exception(
                        "health_check failed for runtime %s",
                        rid,
                    )

    async def stop_one(
        self,
        runtime_id: str,
        *,
        timeout: float = 5.0,
    ) -> None:
        """停止单个 Runtime。"""
        rt = self._runtimes.get(runtime_id)
        if rt is None:
            return
        await rt.stop(timeout=timeout)

    async def shutdown_all(
        self,
        *,
        timeout: float = 10.0,
        per_runtime_timeout: float | None = None,
    ) -> ShutdownReport:
        """停止所有 Runtime，幂等，聚合异常，支持总超时和优先级排序。

        - timeout: 总超时（所有 Runtime 停止的整体时限）。
        - per_runtime_timeout: 单个 Runtime 超时；None 时使用 timeout。
        - 按 priority 升序停止（数值越小越先停止）。
        - 已调用过 shutdown_all 后再次调用为空操作。
        - 单个 Runtime 停止失败不阻止其他 Runtime 的停止。
        """
        if self._stopped:
            return ShutdownReport()
        self._stopped = True

        await self.stop_health_monitor()
        await self.stop_exit_monitor()

        runtimes = list(self._runtimes.values())
        if not runtimes:
            return ShutdownReport()

        prt = per_runtime_timeout or timeout
        sorted_runtimes = sorted(
            runtimes,
            key=lambda rt: self._priorities.get(rt.runtime_id, 0),
        )

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(self._safe_stop(rt, prt) for rt in sorted_runtimes),
                    return_exceptions=False,
                ),
                timeout=timeout,
            )
        except TimeoutError:
            results = [
                await self._safe_stop_no_wait(rt, prt)
                for rt in sorted_runtimes
            ]

        report = ShutdownReport()
        for rt, (status, err) in zip(sorted_runtimes, results, strict=True):
            if status == "ok":
                report.stopped.append(rt.runtime_id)
            elif status == "timeout":
                report.timed_out.append(rt.runtime_id)
            else:
                report.failed[rt.runtime_id] = err or "unknown error"
        if report.has_errors:
            logger.warning(
                "shutdown_all completed with %d failures, %d timeouts: %s",
                len(report.failed),
                len(report.timed_out),
                {"failed": report.failed, "timed_out": report.timed_out},
            )
        return report

    async def stop_all(self, *, timeout: float = 10.0) -> ShutdownReport:
        """向后兼容别名：等价于 shutdown_all。"""
        return await self.shutdown_all(timeout=timeout)

    async def _safe_stop(
        self,
        rt: Any,
        timeout: float,
    ) -> tuple[str, str | None]:
        """安全停止单个 Runtime，返回 (状态, 错误信息)。

        状态: "ok" | "failed" | "timeout"。
        """
        try:
            await asyncio.wait_for(rt.stop(timeout=timeout), timeout=timeout)
            return "ok", None
        except TimeoutError:
            return "timeout", None
        except Exception as exc:
            return "failed", str(exc)

    async def _safe_stop_no_wait(
        self,
        rt: Any,
        timeout: float,
    ) -> tuple[str, str | None]:
        """尝试停止但不等待（用于总超时后的尽力清理）。"""
        try:
            if rt.alive and rt.capabilities.supports_cancel:
                rt.cancel()
            return "timeout", None
        except Exception as exc:
            return "failed", str(exc)

    async def cancel_all(self) -> None:
        """向所有活跃 Runtime 发送取消信号。"""
        for rt in self._runtimes.values():
            if rt.alive and rt.capabilities.supports_cancel:
                with contextlib.suppress(Exception):
                    result = rt.cancel()
                    if asyncio.iscoroutine(result):
                        await result

    def clear_terminal(self) -> list[str]:
        """移除所有已到达终态的 Runtime，返回被移除的 ID 列表。"""
        terminal_ids = [
            rid
            for rid, rt in self._runtimes.items()
            if not rt.alive and rt.status in _TERMINAL_STATUSES
        ]
        for rid in terminal_ids:
            self._runtimes.pop(rid, None)
            self._priorities.pop(rid, None)
            self._exit_monitor_runtimes.discard(rid)
        return terminal_ids

    @property
    def count(self) -> int:
        """已注册 Runtime 数量。"""
        return len(self._runtimes)

    @property
    def active_count(self) -> int:
        """活跃 Runtime 数量。"""
        return sum(1 for rt in self._runtimes.values() if rt.alive)

    @property
    def is_closed(self) -> bool:
        """Coordinator 是否已关闭。"""
        return self._stopped
