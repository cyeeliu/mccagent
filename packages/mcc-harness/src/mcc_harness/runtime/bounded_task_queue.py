"""BoundedTaskQueue：有界异步任务队列，提供背压控制。

当队列满时，put 操作阻塞（背压）或抛出 QueueFullError。
支持优雅关闭：close 后不再接受新任务，drain 排空剩余任务。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class QueueFullError(Exception):
    """队列已满，无法放入任务。"""


class QueueClosedError(Exception):
    """队列已关闭，不再接受新任务。"""


@dataclass
class _QueueEntry:
    task: Any
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future(),
    )


class BoundedTaskQueue:
    """有界异步任务队列。

    max_size 为 0 表示无限制。
    put 在队列满时阻塞，提供背压。
    """

    def __init__(self, *, max_size: int = 0) -> None:
        self._max_size = max_size
        self._queue: asyncio.Queue[_QueueEntry | None] = asyncio.Queue(
            maxsize=max_size if max_size > 0 else 0,
        )
        self._closed = False
        self._pending: int = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def pending(self) -> int:
        return self._pending

    async def put(self, task: Any) -> asyncio.Future:
        """放入任务，返回关联 Future。队列满时阻塞。"""
        if self._closed:
            raise QueueClosedError()
        entry = _QueueEntry(task=task)
        self._pending += 1
        await self._queue.put(entry)
        return entry.future

    def put_nowait(self, task: Any) -> asyncio.Future:
        """非阻塞放入任务。队列满时抛出 QueueFullError。"""
        if self._closed:
            raise QueueClosedError()
        entry = _QueueEntry(task=task)
        self._pending += 1
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull as exc:
            self._pending -= 1
            raise QueueFullError(str(exc)) from exc
        return entry.future

    async def get(self) -> _QueueEntry | None:
        """取出任务条目。队列关闭且空时返回 None。"""
        entry = await self._queue.get()
        if entry is None:
            return None
        return entry

    def task_done(self) -> None:
        """标记一个已取出的任务完成。"""
        self._pending -= 1
        self._queue.task_done()

    async def close(
        self,
        *,
        drain: bool = False,
        num_sentinels: int = 1,
    ) -> list[Any]:
        """关闭队列。drain=True 时排空并返回未处理任务。

        drain=False 时放入 num_sentinels 个 None 哨兵，使每个阻塞在 get()
        的消费者都能收到关闭信号并优雅退出。num_sentinels 应等于消费者
        （worker）数量，否则多余消费者将永久阻塞。
        """
        self._closed = True
        drained: list[Any] = []
        if drain:
            while not self._queue.empty():
                entry = await self._queue.get()
                if entry is not None:
                    drained.append(entry.task)
                    self._pending -= 1
        else:
            for _ in range(max(1, num_sentinels)):
                await self._queue.put(None)
        return drained


__all__ = ["BoundedTaskQueue", "QueueClosedError", "QueueFullError"]
