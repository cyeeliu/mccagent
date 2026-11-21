"""PR-02 修复: PersistentRuntimeV2.cancel_task 必须等待真实执行终止后再发事件。

验证：
- cancel_task 调用后，TASK_CANCELLED 事件必须在真实执行终止后才发出
- 不能在 cancel_token.cancel() 后立即发事件，此时执行可能仍在运行
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeEventType, TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class SlowCancellableExecutor:
    """执行时等待 cancellation，但 cancellation 后仍有收尾工作。"""

    def __init__(self) -> None:
        self._closed = False
        self.started = asyncio.Event()
        self.terminated = asyncio.Event()

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
                # 模拟收尾工作：写入 Store 等
                await asyncio.sleep(0.05)
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


def _make_task() -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="cancel wait termination",
    )


class TestPR02CancelWaitsTermination:
    """PR-02 修复: cancel_task 等待真实执行终止后再发事件。"""

    async def test_cancel_task_awaits_future(self) -> None:
        """cancel_task 应等待 future 完成而非立即返回。

        cancel_task 返回后，真实执行必须已终止。
        如果 cancel_task 不等待，执行终止事件在 cancel_task 返回后才 set。
        """
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = SlowCancellableExecutor()
        rt = PersistentRuntimeV2(executor, agent_id="pr02-future-test")
        await rt.start()

        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()

        # cancel_task 应阻塞直到执行终止
        await rt.cancel_task(task.task_id, task.attempt_id, "test")

        # cancel_task 返回后，执行必须已终止
        assert executor.terminated.is_set(), (
            "cancel_task 返回时执行尚未终止，TASK_CANCELLED 事件可能在执行终止前发出"
        )
        await rt.stop()
