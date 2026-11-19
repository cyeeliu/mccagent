"""WorkerPool：异步工作线程池，从队列消费任务并执行。

每个 worker 循环：从队列取任务 → executor.execute(task) → 设置 future 结果。
支持并发上限（max_workers）、优雅启动/停止。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .bounded_task_queue import BoundedTaskQueue

logger = logging.getLogger(__name__)


class WorkerPool:
    """异步工作线程池。

    从 BoundedTaskQueue 消费任务，通过 executor 执行，
    结果设置到关联 Future。
    """

    def __init__(
        self,
        queue: BoundedTaskQueue,
        executor: Any,
        *,
        max_workers: int = 1,
        on_event: Any | None = None,
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._max_workers = max_workers
        self._on_event = on_event
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._stop_event = asyncio.Event()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def active_workers(self) -> int:
        return sum(1 for w in self._workers if not w.done())

    def start(self) -> None:
        """启动所有 worker。"""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker_loop(i), name=f"worker-{i}")
            self._workers.append(task)

    async def stop(self, *, timeout: float = 5.0) -> None:
        """停止所有 worker，等待当前任务完成。"""
        if not self._started:
            return
        self._stop_event.set()
        # 为每个 worker 放入一个 sentinel，使其都能优雅退出而非依赖超时+cancel
        await self._queue.close(num_sentinels=self._max_workers)
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            for w in self._workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._started = False

    async def _worker_loop(self, worker_id: int) -> None:
        """单个 worker 循环：取任务 → 执行 → 设置结果。"""
        while not self._stop_event.is_set():
            try:
                entry = await self._queue.get()
            except Exception:
                break
            if entry is None:
                break
            task = entry.task
            future = entry.future
            if future.done():
                self._queue.task_done()
                continue
            try:
                if self._on_event is not None:
                    await self._on_event(
                        "task_started",
                        task_id=getattr(task, "task_id", None),
                        attempt_id=getattr(task, "attempt_id", None),
                    )
                result = await self._executor.execute(
                    task,
                    emit=None,
                    cancellation=None,
                )
                if not future.done():
                    future.set_result(result)
                if self._on_event is not None:
                    event_type = "task_completed" if result.is_success else "task_failed"
                    await self._on_event(
                        event_type,
                        task_id=getattr(task, "task_id", None),
                        attempt_id=getattr(task, "attempt_id", None),
                        payload={
                            "summary": result.summary,
                            "error": str(result.error) if result.error else None,
                        },
                    )
            except asyncio.CancelledError:
                if not future.done():
                    future.set_exception(asyncio.CancelledError())
                raise
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
                if self._on_event is not None:
                    await self._on_event(
                        "task_failed",
                        task_id=getattr(task, "task_id", None),
                        attempt_id=getattr(task, "attempt_id", None),
                        payload={"error": str(exc)},
                    )
            finally:
                self._queue.task_done()


__all__ = ["WorkerPool"]
