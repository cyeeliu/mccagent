"""PR-01: TaskRuntime RunningExecution 追踪（P0-01）。

验证：
- cancel_task 后 _execute 在 2s 内退出（等待后台任务完成）
- stop(cancel_running=True) 等待运行中任务完成后再关闭 executor
- _running 按 attempt_id 追踪 asyncio.Task 与 CancellationToken
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeStatus, TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class CancellableExecutor:
    """可取消桩执行器：等待 cancellation 后返回 CANCELLED，追踪执行生命周期。

    close() 内部纯同步不 yield，确保旧代码 stop 不会因 close 让出事件循环，
    从而使红灯测试可靠。
    """

    def __init__(self, *, tail_delay: float = 0.05) -> None:
        self._closed = False
        self._tail_delay = tail_delay
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancel_observed = False

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
                self.cancel_observed = True
                await asyncio.sleep(self._tail_delay)
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
        objective="long running",
    )


class TestPR01TaskRuntimeTracking:
    """PR-01: TaskRuntime 追踪 submit 创建的后台任务。"""

    async def test_cancel_stops_execution(self) -> None:
        """cancel_task 后 _execute 在 2s 内退出。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = CancellableExecutor()
        rt = TaskRuntime(executor, agent_id="pr01-cancel-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()

        t0 = time.monotonic()
        await rt.cancel_task(task.task_id, task.attempt_id, "test cancel")
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"cancel_task 耗时 {elapsed:.2f}s，未在 2s 内退出"
        assert executor.finished.is_set(), "_execute 未在 cancel 后退出"
        assert executor.cancel_observed, "executor 未观测到 cancellation"
        await rt.stop()

    async def test_stop_waits_for_running(self) -> None:
        """stop(cancel_running=True) 等待运行中任务完成。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = CancellableExecutor()
        rt = TaskRuntime(executor, agent_id="pr01-stop-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()

        await rt.stop(cancel_running=True)

        assert executor.finished.is_set(), "stop 未等待运行中任务完成"
        assert rt.status is RuntimeStatus.STOPPED

    async def test_running_cleared_after_completion(self) -> None:
        """任务自然完成后 _running 被清理。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = CancellableExecutor(tail_delay=0.01)
        rt = TaskRuntime(executor, agent_id="pr01-clear-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()
        # 无 cancellation 时 executor 会 sleep(100)，用 cancel 触发完成
        await rt.cancel_task(task.task_id, task.attempt_id)
        # 等待 _running 清理
        await asyncio.sleep(0.05)
        assert task.attempt_id not in rt._running, "_running 未在完成后清理"
        await rt.stop()
