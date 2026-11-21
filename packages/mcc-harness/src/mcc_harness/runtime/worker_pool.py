"""WorkerPool：异步工作线程池，从队列消费任务并执行。

每个 worker 循环：从队列取任务 → executor.execute(task) → 设置 future 结果。
支持并发上限（max_workers）、优雅启动/停止。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from mcc_core.runtime.errors import RuntimeStopTimeoutError

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
        runtime_id: str = "",
    ) -> None:
        self._queue = queue
        self._executor = executor
        self._max_workers = max_workers
        self._on_event = on_event
        self._runtime_id = runtime_id
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._stop_event = asyncio.Event()
        self._cancellations: dict[str, Any] = {}

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

    async def stop(
        self,
        *,
        timeout: float = 5.0,
        drain: bool = True,
        cancel_running: bool = False,
    ) -> None:
        """停止所有 worker，等待当前任务完成。

        生命周期：ACCEPTING → DRAINING → STOPPING → STOPPED。

        - drain=True：现有队列全部执行完成。
        - drain=False：队列中未执行任务逐一标记为取消。
        - cancel_running=True：立即取消正在运行的任务。
        - 所有队列 Future 必须完成或带异常结束。
        - stop 超时后对正在运行任务生成明确结果，不允许静默丢弃。

        若 worker 在超时内未完成，取消所有 worker 并抛出
        RuntimeStopTimeoutError，使调用方感知停止超时。
        """
        if not self._started:
            return
        self._stop_event.set()

        # -- DRAINING 阶段 --
        if drain:
            # 放入哨兵，worker 会处理完所有现有任务后收到哨兵退出
            await self._queue.close(num_sentinels=self._max_workers)
        else:
            # 排空未执行任务，为它们的 future 生成明确取消终态
            self._queue._closed = True
            remaining = await self._queue.drain_pending()
            for entry in remaining:
                if not entry.future.done():
                    entry.future.set_exception(asyncio.CancelledError())
            # 放入哨兵唤醒空闲 worker
            await self._queue.close(num_sentinels=self._max_workers)

        # -- STOPPING 阶段 --
        if cancel_running:
            # 立即取消所有正在运行的 worker
            for w in self._workers:
                if not w.done():
                    w.cancel()

        timed_out = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            timed_out = True
            for w in self._workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)

        # -- STOPPED 阶段：确保所有 future 有明确终态 --
        leftover = await self._queue.drain_pending()
        for entry in leftover:
            if not entry.future.done():
                if timed_out:
                    entry.future.set_exception(
                        RuntimeStopTimeoutError(self._runtime_id, timeout),
                    )
                else:
                    entry.future.set_exception(asyncio.CancelledError())

        self._workers.clear()
        self._started = False
        if timed_out:
            raise RuntimeStopTimeoutError(self._runtime_id, timeout)

    async def _worker_loop(self, worker_id: int) -> None:
        """单个 worker 循环：取任务 → 执行 → 设置结果。

        使用 while True 依赖哨兵（None）退出，确保 stop(drain=True) 时
        worker 会处理完队列中所有任务再退出。
        """
        while True:
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
            attempt_id = getattr(task, "attempt_id", None)
            cancellation = self._cancellations.get(attempt_id) if attempt_id else None
            try:
                if self._on_event is not None:
                    await self._on_event(
                        "task_started",
                        task_id=getattr(task, "task_id", None),
                        attempt_id=attempt_id,
                    )
                timeout = getattr(task, "timeout_seconds", None)
                if timeout is not None:
                    result = await asyncio.wait_for(
                        self._executor.execute(
                            task,
                            emit=None,
                            cancellation=cancellation,
                        ),
                        timeout=timeout,
                    )
                else:
                    result = await self._executor.execute(
                        task,
                        emit=None,
                        cancellation=cancellation,
                    )
                if not future.done():
                    future.set_result(result)
                if self._on_event is not None:
                    from mcc_harness.runtime.event_payload import (
                        build_result_payload,
                        result_event_type,
                    )

                    event_type = result_event_type(result)
                    task_id = getattr(task, "task_id", None)
                    attempt_id = getattr(task, "attempt_id", None)
                    payload = build_result_payload(
                        task_id or "",
                        attempt_id or "",
                        result,
                    )
                    await self._on_event(
                        event_type.value,
                        task_id=task_id,
                        attempt_id=attempt_id,
                        payload=payload,
                    )
            except TimeoutError:
                from mcc_core.runtime.errors import TaskExecutionTimeoutError

                exc = TaskExecutionTimeoutError(
                    attempt_id or "",
                    timeout or 0.0,
                )
                if not future.done():
                    future.set_exception(exc)
                if cancellation is not None:
                    cancellation.cancel("execution timeout")
                if self._on_event is not None:
                    await self._on_event(
                        "task_timed_out",
                        task_id=getattr(task, "task_id", None),
                        attempt_id=attempt_id,
                        payload={"timeout_seconds": timeout},
                    )
            except asyncio.CancelledError:
                if not future.done():
                    future.set_exception(asyncio.CancelledError())
                if self._on_event is not None:
                    with contextlib.suppress(Exception):
                        await self._on_event(
                            "task_cancelled",
                            task_id=getattr(task, "task_id", None),
                            attempt_id=attempt_id,
                            payload={"reason": "worker cancelled"},
                        )
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
                if attempt_id is not None:
                    self._cancellations.pop(attempt_id, None)
                self._queue.task_done()


__all__ = ["WorkerPool"]
