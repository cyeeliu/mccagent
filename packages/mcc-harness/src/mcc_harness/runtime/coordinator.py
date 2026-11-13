"""RuntimeCoordinator：运行时生命周期协调器。

统一管理 Runtime 的注册、注销、查询、停止和关闭清理。
提供活跃 Runtime 列表和状态快照，聚合关闭异常。
不执行规划任务或选择 Agent，仅负责生命周期管理。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from .base import AgentRuntime, RuntimeStatus

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSnapshot:
    """运行时状态快照。"""

    runtime_id: str
    status: RuntimeStatus
    alive: bool


@dataclass
class ShutdownReport:
    """关闭报告：汇总 stop_all 的结果和异常。"""

    stopped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return bool(self.failed)

    @property
    def total(self) -> int:
        return len(self.stopped) + len(self.failed)


class RuntimeCoordinator:
    """运行时生命周期协调器。

    职责：
    - Runtime 注册、注销和查询（重复 ID 抛 RuntimeError）。
    - 活跃 Runtime 列表和状态快照。
    - stop_all 统一关闭，幂等，聚合异常。
    - 终态 Runtime 归档（可选 TTL 清理）。

    不负责规划任务或选择 Agent。
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, AgentRuntime] = {}
        self._lock = asyncio.Lock()
        self._stopped = False

    def register(self, runtime: AgentRuntime) -> None:
        """注册 Runtime，重复 ID 抛 RuntimeError。"""
        rid = runtime.runtime_id
        if rid in self._runtimes:
            raise RuntimeError(f"duplicate runtime id: {rid}")
        self._runtimes[rid] = runtime

    def unregister(self, runtime_id: str) -> AgentRuntime | None:
        """注销 Runtime，返回被移除的实例（不存在则返回 None）。"""
        return self._runtimes.pop(runtime_id, None)

    def get(self, runtime_id: str) -> AgentRuntime | None:
        """按 ID 查询 Runtime（不存在返回 None）。"""
        return self._runtimes.get(runtime_id)

    def list_active(self) -> list[AgentRuntime]:
        """返回所有活跃（alive）的 Runtime。"""
        return [rt for rt in self._runtimes.values() if rt.alive]

    def list_all(self) -> list[AgentRuntime]:
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

    async def stop_all(self, *, timeout: float = 10.0) -> ShutdownReport:
        """停止所有 Runtime，幂等，聚合异常。

        已调用过 stop_all 后再次调用为空操作。
        单个 Runtime 停止失败不阻止其他 Runtime 的停止。
        """
        if self._stopped:
            return ShutdownReport()
        self._stopped = True

        runtimes = list(self._runtimes.values())
        if not runtimes:
            return ShutdownReport()

        per_timeout = max(timeout / len(runtimes), 0.5)
        results = await asyncio.gather(
            *(self._safe_stop(rt, per_timeout) for rt in runtimes),
            return_exceptions=False,
        )

        report = ShutdownReport()
        for rt, (ok, err) in zip(runtimes, results, strict=True):
            if ok:
                report.stopped.append(rt.runtime_id)
            else:
                report.failed[rt.runtime_id] = err
        if report.has_errors:
            logger.warning(
                "stop_all completed with %d failures: %s",
                len(report.failed),
                report.failed,
            )
        return report

    async def _safe_stop(
        self,
        rt: AgentRuntime,
        timeout: float,
    ) -> tuple[bool, str | None]:
        """安全停止单个 Runtime，返回 (成功, 错误信息)。"""
        try:
            await rt.stop(timeout=timeout)
            return True, None
        except Exception as exc:
            return False, str(exc)

    async def cancel_all(self) -> None:
        """向所有活跃 Runtime 发送取消信号。"""
        for rt in self._runtimes.values():
            if rt.alive:
                with contextlib.suppress(Exception):
                    rt.cancel()

    def clear_terminal(self) -> list[str]:
        """移除所有已到达终态的 Runtime，返回被移除的 ID 列表。"""
        terminal_ids = [
            rid
            for rid, rt in self._runtimes.items()
            if not rt.alive
            and rt.status
            in (
                RuntimeStatus.COMPLETED,
                RuntimeStatus.FAILED,
                RuntimeStatus.CANCELLED,
                RuntimeStatus.TIMEOUT,
                RuntimeStatus.SHUTDOWN,
            )
        ]
        for rid in terminal_ids:
            self._runtimes.pop(rid, None)
        return terminal_ids

    @property
    def count(self) -> int:
        """已注册 Runtime 数量。"""
        return len(self._runtimes)

    @property
    def active_count(self) -> int:
        """活跃 Runtime 数量。"""
        return sum(1 for rt in self._runtimes.values() if rt.alive)
