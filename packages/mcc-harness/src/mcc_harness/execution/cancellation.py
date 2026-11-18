"""CancellationToken 实现。

协作式取消信号，基于 asyncio.Event。
AgentExecutor.execute 通过 CancellationToken 接收取消请求，
在下一安全边界（迭代间）生效。
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str | None = None

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str | None = None) -> None:
        self._reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def to_asyncio_event(self) -> asyncio.Event:
        return self._event


__all__ = ["CancellationToken"]
