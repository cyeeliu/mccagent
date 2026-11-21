"""PR-03 修复: WorkerPool CancelledError 分支必须发 task_cancelled 事件。

验证：
- 当 worker task 被外部取消（stop(cancel_running=True)）时，
  WorkerPool._worker_loop 的 CancelledError 分支必须发 task_cancelled 事件，
  而非静默丢弃。
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class SlowExecutor:
    """执行时长时间阻塞的桩执行器，等待外部取消。"""

    def __init__(self) -> None:
        self._closed = False
        self.started = asyncio.Event()

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.started.set()
        # 长时间阻塞，等待被外部 cancel
        await asyncio.sleep(100)
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="stub",
            agent_id=task.agent_id,
            status="succeeded",
            summary="done",
        )

    async def close(self) -> None:
        self._closed = True


def _make_task() -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="worker cancel event",
    )


class TestPR03WorkerCancelledEvent:
    """PR-03 修复: WorkerPool CancelledError 分支发 task_cancelled 事件。"""

    async def test_cancelled_error_emits_task_cancelled_event(self) -> None:
        """worker 被外部取消时必须发 task_cancelled 事件。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        received_events: list[tuple[str, str | None, str | None]] = []

        async def on_event(
            event_type: str,
            *,
            task_id: str | None = None,
            attempt_id: str | None = None,
            payload: dict[str, object] | None = None,
        ) -> None:
            received_events.append((event_type, task_id, attempt_id))

        executor = SlowExecutor()
        queue = BoundedTaskQueue(max_size=0)
        pool = WorkerPool(
            queue,
            executor,
            max_workers=1,
            on_event=on_event,
            runtime_id="pr03-test",
        )
        pool.start()
        task = _make_task()
        await queue.put(task)
        await executor.started.wait()

        # 取消正在运行的 worker
        await pool.stop(cancel_running=True, drain=False, timeout=3.0)

        event_types = [e[0] for e in received_events]
        assert "task_cancelled" in event_types, (
            f"CancelledError 分支未发 task_cancelled 事件，收到的事件: {event_types}"
        )
