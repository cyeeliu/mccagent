"""Phase 4 深度修复验证：WorkerPool 关闭语义（P1-03）。

P1-03: WorkerPool 关闭语义有缺陷：
       - drain=True 时所有排队任务都应被执行
       - drain=False 时未执行任务 future 应被标记为取消
       - stop 后所有 future 必须完成，不允许遗留
       - cancel_running=True 时正在执行的任务应被立即取消
       - PersistentRuntimeV2.stop(cancel_running=...) 应传递给 WorkerPool
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeResult, TaskStatus, TokenUsage
from mcc_core.runtime.task import RuntimeTask


class StubExecutor:
    """桩 Executor：可配置延迟，返回完整 RuntimeResult。"""

    def __init__(self, *, delay: float = 0.01) -> None:
        self._delay = delay
        self._closed = False
        self.executed: list[str] = []

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> Any:
        await asyncio.sleep(self._delay)
        self.executed.append(task.task_id)
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="stub",
            agent_id=task.agent_id,
            status=TaskStatus.SUCCEEDED,
            summary=f"done: {task.objective}",
            output={"answer": 42},
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            cost_usd=0.001,
        )

    async def close(self) -> None:
        self._closed = True


def _make_task(
    *,
    task_id: str | None = None,
    attempt_id: str | None = None,
    agent_id: str = "test-agent",
    objective: str = "test objective",
) -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id or f"task-{uuid4().hex[:8]}",
        attempt_id=attempt_id or f"att-{uuid4().hex[:8]}",
        agent_id=agent_id,
        objective=objective,
    )


# -- P1-03: WorkerPool drain 语义 --


class TestWorkerPoolDrainSemantics:
    """验证 WorkerPool.stop(drain=True) 执行所有排队任务。"""

    async def test_drain_true_completes_all_queued_tasks(self):
        """drain=True 时所有排队任务都应被执行，future 全部完成。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.1)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        # 放入 3 个任务
        futures: list[asyncio.Future] = []
        for i in range(3):
            task = _make_task(objective=f"task-{i}")
            f = await queue.put(task)
            futures.append(f)

        # 等待第一个任务被取出开始执行
        await asyncio.sleep(0.05)

        # drain=True 停止，应完成所有任务
        await pool.stop(timeout=5.0, drain=True)

        # 所有 future 都应完成
        for i, f in enumerate(futures):
            assert f.done(), f"future[{i}] 未完成：drain=True 应执行所有排队任务"

    async def test_drain_true_executes_all_with_multiple_workers(self):
        """max_workers=2, drain=True 时所有任务都应被执行。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.1)
        pool = WorkerPool(queue, executor, max_workers=2)
        pool.start()

        futures: list[asyncio.Future] = []
        for i in range(5):
            task = _make_task(objective=f"task-{i}")
            f = await queue.put(task)
            futures.append(f)

        await asyncio.sleep(0.05)
        await pool.stop(timeout=5.0, drain=True)

        for i, f in enumerate(futures):
            assert f.done(), f"future[{i}] 未完成：drain=True 应执行所有任务"


# -- P1-03: WorkerPool drain=False 语义 --


class TestWorkerPoolNoDrainSemantics:
    """验证 WorkerPool.stop(drain=False) 为未执行任务生成明确终态。"""

    async def test_drain_false_resolves_all_futures(self):
        """drain=False 时所有 future 必须完成（执行完的或被取消的）。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.3)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        futures: list[asyncio.Future] = []
        for i in range(4):
            task = _make_task(objective=f"task-{i}")
            f = await queue.put(task)
            futures.append(f)

        # 等待第一个任务开始执行
        await asyncio.sleep(0.05)

        # drain=False 停止
        await pool.stop(timeout=5.0, drain=False)

        # 所有 future 都应完成
        for i, f in enumerate(futures):
            assert f.done(), f"future[{i}] 未完成：drain=False 应为所有未执行任务生成明确终态"

    async def test_drain_false_unexecuted_futures_have_exception(self):
        """drain=False 时未执行任务的 future 应带异常完成（非成功）。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.5)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        futures: list[asyncio.Future] = []
        for i in range(3):
            task = _make_task(objective=f"task-{i}")
            f = await queue.put(task)
            futures.append(f)

        # 等待第一个任务开始执行
        await asyncio.sleep(0.05)

        await pool.stop(timeout=5.0, drain=False)

        # 至少有一些 future 应带异常完成（未被执行的任务）
        exception_count = 0
        for f in futures:
            if f.done() and f.exception() is not None:
                exception_count += 1
        assert exception_count >= 1, "drain=False 应至少有一个未执行任务带异常完成"


# -- P1-03: WorkerPool 无遗留未完成 future --


class TestNoUnresolvedFutures:
    """验证 stop 后所有 future 必须完成。"""

    @pytest.mark.parametrize("max_workers", [1, 2, 4])
    async def test_no_unresolved_futures_after_stop(self, max_workers):
        """stop 后不允许遗留未完成 future。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.05)
        pool = WorkerPool(queue, executor, max_workers=max_workers)
        pool.start()

        futures: list[asyncio.Future] = []
        for i in range(max_workers * 2 + 1):
            task = _make_task(objective=f"task-{i}")
            f = await queue.put(task)
            futures.append(f)

        await asyncio.sleep(0.02)
        await pool.stop(timeout=5.0)

        for i, f in enumerate(futures):
            assert f.done(), f"future[{i}] 未完成：stop 后不允许遗留未完成 future"


# -- P1-03: WorkerPool cancel_running 语义 --


class TestWorkerPoolCancelRunning:
    """验证 cancel_running=True 立即取消运行中任务。"""

    async def test_cancel_running_cancels_executing_task(self):
        """cancel_running=True 时正在执行的任务应被取消，stop 快速完成。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=2.0)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        task = _make_task()
        f = await queue.put(task)

        # 等待任务开始执行
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await pool.stop(timeout=5.0, cancel_running=True)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"stop 耗时 {elapsed:.3f}s，cancel_running=True 应立即取消"
        assert f.done(), "cancel_running=True 应使正在执行任务的 future 完成"

    async def test_cancel_running_false_waits_for_task(self):
        """cancel_running=False 时应等待任务完成。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = StubExecutor(delay=0.3)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        task = _make_task()
        f = await queue.put(task)

        await asyncio.sleep(0.05)
        await pool.stop(timeout=5.0, cancel_running=False)

        assert f.done(), "cancel_running=False 应等待任务完成"
        # 任务应成功完成（非异常）
        assert f.exception() is None, "任务应成功完成"


# -- P1-03: PersistentRuntimeV2 cancel_running 传递 --


class TestPersistentRuntimeCancelRunning:
    """验证 PersistentRuntimeV2.stop(cancel_running=...) 传递给 WorkerPool。"""

    async def test_persistent_stop_cancel_running_is_immediate(self):
        """PersistentRuntimeV2.stop(cancel_running=True) 应立即取消运行中任务。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = StubExecutor(delay=2.0)
        rt = PersistentRuntimeV2(executor, max_concurrency=1, agent_id="test")
        await rt.start()

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await rt.stop(timeout=5.0, cancel_running=True)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"stop 耗时 {elapsed:.3f}s，cancel_running=True 应立即取消"

    async def test_persistent_stop_cancel_running_false_waits(self):
        """PersistentRuntimeV2.stop(cancel_running=False) 应等待任务完成。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = StubExecutor(delay=0.3)
        rt = PersistentRuntimeV2(executor, max_concurrency=1, agent_id="test")
        await rt.start()

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.05)

        await rt.stop(timeout=5.0, cancel_running=False)

        f = rt._task_futures.get(task.task_id)
        assert f is not None
        assert f.done(), "cancel_running=False 应等待任务完成"
