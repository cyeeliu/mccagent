"""应用生命周期管理：统一 shutdown 清理。

AgentApplication 持有 RuntimeCoordinator，在应用退出时
确保所有 Runtime 干净关闭，无遗留任务。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentApplication:
    """应用生命周期管理器。

    持有 RuntimeCoordinator，提供统一 shutdown 入口。
    可作为 async context manager 使用。
    """

    def __init__(self, coordinator: Any | None = None) -> None:
        self._coordinator = coordinator
        self._shutdown = False

    @classmethod
    def from_source(cls, source: Any) -> AgentApplication:
        """从 AgentEventSource 提取 coordinator 构建应用实例。"""
        coordinator = getattr(source, "_coordinator", None)
        return cls(coordinator=coordinator)

    @property
    def coordinator(self) -> Any | None:
        return self._coordinator

    async def shutdown(self, *, timeout: float = 10.0) -> None:
        """关闭所有 Runtime，幂等，聚合异常不抛出。"""
        if self._shutdown or self._coordinator is None:
            return
        self._shutdown = True
        report = await self._coordinator.stop_all(timeout=timeout)
        if report.has_errors:
            logger.warning(
                "shutdown completed with %d failure(s): %s",
                len(report.failed),
                report.failed,
            )

    async def __aenter__(self) -> AgentApplication:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.shutdown()
