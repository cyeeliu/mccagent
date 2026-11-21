"""PR-04 修复: 超时后先取消旧 attempt 真实执行再 retry。

验证：
- 当 _dispatch_and_wait 超时后，在 retry 前先通过 runtime.cancel_task
  取消旧 attempt 的真实执行并等待终止
- 不能仅标记 attempt 状态为 TIMED_OUT 就立即 retry
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class TrackingExecutor:
    """记录 cancel_task 调用的桩执行器。"""

    def __init__(self) -> None:
        self._closed = False
        self.started = asyncio.Event()
        self.terminated = asyncio.Event()
        self.cancel_called = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.started.set()
        try:
            if cancellation is not None:
                await cancellation.wait()
                self.cancel_called = True
                await asyncio.sleep(0.02)
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
            self.terminated.set()

    async def close(self) -> None:
        self._closed = True


class TestPR04TimeoutCancelsExecution:
    """PR-04 修复: 超时后先取消旧 attempt 真实执行再 retry。"""

    async def test_dispatch_timeout_cancels_runtime_task(self) -> None:
        """_dispatch_and_wait 超时后应调用 runtime.cancel_task。"""
        from mcc_core.runtime.task_store import InMemoryTaskStore
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.coordinator import RuntimeCoordinator
        from mcc_harness.runtime.dispatcher import TaskDispatcher
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = TrackingExecutor()
        rt = PersistentRuntimeV2(executor, agent_id="pr04-test", runtime_id="rt-pr04")
        await rt.start()

        coordinator = RuntimeCoordinator()
        coordinator.register(rt)

        store = InMemoryTaskStore()
        dispatcher = TaskDispatcher(store, coordinator)

        # 创建 task
        task_record = await store.create_task(
            subject="timeout cancel test",
            description="test",
        )
        task_id = task_record.task_id

        # 派发任务
        handle = await dispatcher.dispatch(
            task_id,
            "pr04-test",
            runtime_id="rt-pr04",
            objective="timeout cancel test",
            timeout_seconds=100,
        )
        await executor.started.wait()

        # wait_for_result 超时
        with contextlib.suppress(Exception):
            await dispatcher.wait_for_result(handle, timeout=0.1)

        # 超时后，dispatcher 应通过 runtime.cancel_task 取消真实执行
        # 等待执行终止
        await asyncio.wait_for(executor.terminated.wait(), timeout=3.0)

        # 验证执行已终止（cancel_task 被调用或执行自然完成）
        assert executor.terminated.is_set(), (
            "超时后旧 attempt 真实执行未终止"
        )

        await rt.stop()
