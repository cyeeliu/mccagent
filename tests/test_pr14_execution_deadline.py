"""PR-14: 统一 ExecutionDeadline（P1-04）。

验证：
- timeout_seconds 到期时 Runtime 取消执行，Attempt 写为 TIMED_OUT，发 TASK_TIMED_OUT 事件
- timeout_seconds=None 时不超时，任务正常完成
- 对 TaskRuntime / InteractiveRuntimeV2 / WorkerPool 三种 Runtime 均生效
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeEventType, TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class SlowExecutor:
    """执行耗时 delay 秒后返回 SUCCEEDED。"""

    def __init__(self, delay: float = 2.0) -> None:
        self.delay = delay
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.started.set()
        try:
            await asyncio.sleep(self.delay)
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
        pass


class RecordingTaskStore:
    """记录 fail_attempt / complete_attempt 调用的测试桩。"""

    def __init__(self) -> None:
        self.fail_calls: list[tuple[str, str, Any]] = []
        self.complete_calls: list[str] = []

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        status: Any = None,
    ) -> None:
        from mcc_core.runtime.task_record import AttemptStatus

        self.fail_calls.append((attempt_id, error, status or AttemptStatus.FAILED))

    async def complete_attempt(
        self,
        attempt_id: str,
        **kwargs: Any,
    ) -> None:
        self.complete_calls.append(attempt_id)


def _make_task(*, timeout_seconds: float | None = None) -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="test execution deadline",
        timeout_seconds=timeout_seconds,
    )


async def _collect_events_until(
    runtime: Any,
    target: RuntimeEventType,
    *,
    timeout: float = 5.0,
) -> list[Any]:
    """收集事件直到看到 target 类型或超时。"""
    events: list[Any] = []

    async def collector() -> None:
        async for ev in runtime.subscribe_events():
            events.append(ev)
            if ev.event_type is target:
                break

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(collector(), timeout=timeout)
    return events


class TestPR14TaskRuntimeDeadline:
    """PR-14: TaskRuntime 执行超时。"""

    async def test_timeout_emits_timed_out_event(self) -> None:
        """timeout_seconds 到期时发 TASK_TIMED_OUT 事件，Attempt 写 TIMED_OUT。"""
        from mcc_core.runtime.task_record import AttemptStatus
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = SlowExecutor(delay=2.0)
        store = RecordingTaskStore()
        rt = TaskRuntime(executor, agent_id="pr14-tr-timeout", task_store=store)
        await rt.start()
        task = _make_task(timeout_seconds=0.2)

        collect_task = asyncio.create_task(
            _collect_events_until(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=2.0),
        )
        await rt.submit(task)

        t0 = time.monotonic()
        events = await collect_task
        elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"超时事件未及时到达，耗时 {elapsed:.2f}s"
        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_TIMED_OUT in types, (
            f"未收到 TASK_TIMED_OUT 事件，收到: {types}"
        )
        assert len(store.fail_calls) == 1, f"store.fail_calls={store.fail_calls}"
        aid, _error, status = store.fail_calls[0]
        assert aid == task.attempt_id
        assert status is AttemptStatus.TIMED_OUT, f"status={status}"
        await rt.stop()

    async def test_no_timeout_when_none(self) -> None:
        """timeout_seconds=None 时不超时，任务正常完成。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = SlowExecutor(delay=0.05)
        store = RecordingTaskStore()
        rt = TaskRuntime(executor, agent_id="pr14-tr-none", task_store=store)
        await rt.start()
        task = _make_task(timeout_seconds=None)

        collect_task = asyncio.create_task(
            _collect_events_until(rt, RuntimeEventType.TASK_COMPLETED, timeout=2.0),
        )
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_COMPLETED in types, (
            f"未收到 TASK_COMPLETED 事件，收到: {types}"
        )
        assert len(store.complete_calls) == 1
        assert store.complete_calls[0] == task.attempt_id
        await rt.stop()

    async def test_timeout_does_not_wait_full_delay(self) -> None:
        """超时后不等待 executor 全部 delay，在 timeout 附近返回。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = SlowExecutor(delay=5.0)
        rt = TaskRuntime(executor, agent_id="pr14-tr-fast")
        await rt.start()
        task = _make_task(timeout_seconds=0.1)

        collect_task = asyncio.create_task(
            _collect_events_until(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=3.0),
        )
        t0 = time.monotonic()
        await rt.submit(task)
        events = await collect_task
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"超时后仍等待 {elapsed:.2f}s，未快速返回"
        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_TIMED_OUT in types
        await rt.stop()


class TestPR14InteractiveRuntimeDeadline:
    """PR-14: InteractiveRuntimeV2 执行超时。"""

    async def test_timeout_emits_timed_out_event(self) -> None:
        """timeout_seconds 到期时发 TASK_TIMED_OUT 事件。"""
        from mcc_core.runtime.task_record import AttemptStatus
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        rt = InteractiveRuntimeV2(
            object(),
            runtime_id="pr14-ir-timeout",
            agent_id="pr14-ir-timeout",
            task_store=RecordingTaskStore(),
        )
        rt._executor = SlowExecutor(delay=2.0)
        store = rt._task_store
        await rt.start()
        task = _make_task(timeout_seconds=0.2)

        collect_task = asyncio.create_task(
            _collect_events_until(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=2.0),
        )
        await rt.submit(task)

        t0 = time.monotonic()
        events = await collect_task
        elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"超时事件未及时到达，耗时 {elapsed:.2f}s"
        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_TIMED_OUT in types, (
            f"未收到 TASK_TIMED_OUT 事件，收到: {types}"
        )
        assert len(store.fail_calls) == 1
        aid, _error, status = store.fail_calls[0]
        assert aid == task.attempt_id
        assert status is AttemptStatus.TIMED_OUT, f"status={status}"
        await rt.stop()

    async def test_no_timeout_when_none(self) -> None:
        """timeout_seconds=None 时不超时，任务正常完成。"""
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        rt = InteractiveRuntimeV2(
            object(),
            runtime_id="pr14-ir-none",
            agent_id="pr14-ir-none",
            task_store=RecordingTaskStore(),
        )
        rt._executor = SlowExecutor(delay=0.05)
        store = rt._task_store
        await rt.start()
        task = _make_task(timeout_seconds=None)

        collect_task = asyncio.create_task(
            _collect_events_until(rt, RuntimeEventType.TASK_COMPLETED, timeout=2.0),
        )
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_COMPLETED in types
        assert len(store.complete_calls) == 1
        await rt.stop()


class TestPR14WorkerPoolDeadline:
    """PR-14: WorkerPool 执行超时。"""

    async def test_timeout_sets_future_exception(self) -> None:
        """timeout_seconds 到期时 future 收到 TaskExecutionTimeoutError。"""
        from mcc_core.runtime.errors import TaskExecutionTimeoutError
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = SlowExecutor(delay=5.0)
        emitted: list[tuple[str, str | None]] = []

        async def on_event(
            event_type: str,
            *,
            task_id: str | None = None,
            attempt_id: str | None = None,
            **kw: Any,
        ) -> None:
            emitted.append((event_type, attempt_id))

        pool = WorkerPool(
            queue,
            executor,
            max_workers=1,
            on_event=on_event,
            runtime_id="pr14-wp-test",
        )
        pool.start()

        task = _make_task(timeout_seconds=0.2)
        future = await queue.put(task)

        t0 = time.monotonic()
        with pytest.raises(TaskExecutionTimeoutError):
            await asyncio.wait_for(future, timeout=3.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.5, f"超时后仍等待 {elapsed:.2f}s"
        assert any(et == "task_timed_out" for et, _ in emitted), (
            f"未发出 task_timed_out 事件，收到: {emitted}"
        )
        await pool.stop(timeout=3.0)

    async def test_no_timeout_when_none(self) -> None:
        """timeout_seconds=None 时不超时，任务正常完成。"""
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        queue = BoundedTaskQueue()
        executor = SlowExecutor(delay=0.05)

        pool = WorkerPool(
            queue,
            executor,
            max_workers=1,
            runtime_id="pr14-wp-none",
        )
        pool.start()

        task = _make_task(timeout_seconds=None)
        future = await queue.put(task)
        result = await asyncio.wait_for(future, timeout=2.0)

        assert result.status is TaskStatus.SUCCEEDED
        await pool.stop(timeout=3.0)
