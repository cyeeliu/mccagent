"""PR-02: PersistentRuntimeV2 attempt 级取消（P0-02）。

验证：
- cancel_task 将取消信号传播到 executor.execute 的 cancellation 参数
- WorkerPool._worker_loop 从 _cancellations 取 token 传入 executor
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class RecordingExecutor:
    """记录 cancellation 状态的桩执行器：等待 cancellation 后返回 CANCELLED。"""

    def __init__(self) -> None:
        self._closed = False
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancel_seen = False
        self.cancellation_received: Any | None = None

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.started.set()
        self.cancellation_received = cancellation
        try:
            if cancellation is not None:
                await cancellation.wait()
                self.cancel_seen = True
                return RuntimeResult(
                    task_id=task.task_id,
                    attempt_id=task.attempt_id,
                    runtime_id="stub",
                    agent_id=task.agent_id,
                    status=TaskStatus.CANCELLED,
                    summary="cancelled",
                )
            await asyncio.sleep(100)
            return RuntimeResult(
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                runtime_id="stub",
                agent_id=task.agent_id,
                status=TaskStatus.SUCCEEDED,
                summary="done",
            )
        finally:
            self.finished.set()

    async def close(self) -> None:
        self._closed = True


def _make_task() -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="persistent cancel",
    )


class TestPR02PersistentCancelPropagation:
    """PR-02: PersistentRuntimeV2 cancel 传播到 executor。"""

    async def test_cancel_propagates_to_executor(self) -> None:
        """cancel_task 后 executor 观测到 cancellation.is_cancelled。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = RecordingExecutor()
        rt = PersistentRuntimeV2(executor, agent_id="pr02-cancel-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()

        await rt.cancel_task(task.task_id, task.attempt_id, "test cancel")

        await asyncio.wait_for(executor.finished.wait(), timeout=3.0)
        assert executor.cancel_seen, "executor 未观测到 cancellation"
        assert executor.cancellation_received is not None, "executor 未收到 cancellation 参数"
        await rt.stop()

    async def test_worker_pool_passes_cancellation(self) -> None:
        """WorkerPool._worker_loop 将 cancellation 传入 executor.execute。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        executor = RecordingExecutor()
        queue = BoundedTaskQueue(max_size=0)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()
        task = _make_task()

        from mcc_harness.execution.cancellation import CancellationToken

        token = CancellationToken()
        pool._cancellations[task.attempt_id] = token
        await queue.put(task)
        await executor.started.wait()

        token.cancel()
        await asyncio.wait_for(executor.finished.wait(), timeout=3.0)
        assert executor.cancel_seen, "WorkerPool 未将 cancellation 传入 executor"
        await pool.stop()
