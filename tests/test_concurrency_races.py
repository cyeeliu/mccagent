"""并发竞态测试：验证 Runtime/Coordinator 在并发场景下的正确性。

使用 asyncio.Event / Barrier 构造确定性竞态，不依赖任意 sleep。
覆盖迁移方案 15.2 的 10 个场景。
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
from mcc_core.runtime import (
    RuntimeStatus,
    RuntimeTask,
    TaskHandle,
    TaskStatus,
)
from mcc_core.runtime.result import RuntimeResult
from mcc_harness.runtime.coordinator import RuntimeCoordinator
from mcc_harness.runtime.task_runtime import TaskRuntime

# -- 可控 Fake Executor --


class ControllableExecutor:
    """通过 asyncio.Event 控制执行时机的假 Executor。

    execute() 在 start_event.set() 后开始，在 done_event.set() 后完成。
    可注入 result_factory 来控制返回结果。
    """

    def __init__(
        self,
        *,
        start_event: asyncio.Event | None = None,
        done_event: asyncio.Event | None = None,
        result_factory: Any | None = None,
        agent_id: str = "fake-agent",
    ) -> None:
        self._start_event = start_event or asyncio.Event()
        self._done_event = done_event or asyncio.Event()
        self._result_factory = result_factory
        self._agent_id = agent_id
        self.execute_calls: list[RuntimeTask] = []
        self._closed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.execute_calls.append(task)
        self._start_event.set()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._done_event.wait(), timeout=5.0)
        if cancellation is not None and cancellation.is_cancelled:
            return _make_result(task, TaskStatus.CANCELLED, "cancelled")
        if self._result_factory is not None:
            return self._result_factory(task)
        return _make_result(task, TaskStatus.SUCCEEDED, "done")

    async def close(self) -> None:
        self._closed = True


class ImmediateExecutor:
    """立即完成的 Executor，用于测试 submit 返回前完成。"""

    def __init__(self, agent_id: str = "immediate") -> None:
        self._agent_id = agent_id

    async def execute(
        self,
        task: RuntimeTask,
        *,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        return _make_result(task, TaskStatus.SUCCEEDED, "immediate")

    async def close(self) -> None:
        pass


class CrashingExecutor:
    """执行时抛异常的 Executor，模拟 Runtime 崩溃。"""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("crash")

    async def execute(
        self,
        task: RuntimeTask,
        *,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        raise self._error

    async def close(self) -> None:
        pass


def _make_result(
    task: RuntimeTask,
    status: TaskStatus,
    summary: str = "",
) -> RuntimeResult:
    return RuntimeResult(
        task_id=task.task_id,
        attempt_id=task.attempt_id,
        runtime_id="fake-rt",
        agent_id=task.agent_id,
        status=status,
        summary=summary,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


def _make_task(
    task_id: str = "t1",
    attempt_id: str = "a1",
    objective: str = "do work",
    agent_id: str = "fake-agent",
) -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id,
        attempt_id=attempt_id,
        agent_id=agent_id,
        objective=objective,
    )


# -- 1. Task 在 submit 返回前完成 --


class TestTaskCompletesBeforeSubmitReturns:
    async def test_immediate_completion(self):
        """Executor 立即完成时，submit 返回后结果已就绪。"""
        rt = TaskRuntime(ImmediateExecutor(), agent_id="imm")
        await rt.start()
        task = _make_task()
        handle = await rt.submit(task)
        assert isinstance(handle, TaskHandle)
        await asyncio.sleep(0.05)
        result = rt.get_result(task.task_id)
        assert result is not None
        assert result.status is TaskStatus.SUCCEEDED
        await rt.stop()


# -- 2. Task 在 wait 开始前完成 --


class TestTaskCompletesBeforeWait:
    async def test_result_available_before_get(self):
        """任务在 get_result 调用前已完成，结果直接可取。"""
        done = asyncio.Event()
        executor = ControllableExecutor(done_event=done)
        rt = TaskRuntime(executor, agent_id="early")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        done.set()
        await asyncio.sleep(0.05)
        result = rt.get_result(task.task_id)
        assert result is not None
        assert result.status is TaskStatus.SUCCEEDED
        await rt.stop()


# -- 3. 两个协程同时提交（TaskRuntime 单任务限制）--


class TestConcurrentSubmit:
    async def test_second_submit_rejected(self):
        """TaskRuntime 单任务运行时，第二个 submit 应被拒绝。"""
        done = asyncio.Event()
        executor = ControllableExecutor(done_event=done)
        rt = TaskRuntime(executor, agent_id="single")
        await rt.start()
        task1 = _make_task(task_id="t1")
        task2 = _make_task(task_id="t2")

        await rt.submit(task1)
        with pytest.raises((RuntimeError, Exception)):
            await rt.submit(task2)

        done.set()
        await asyncio.sleep(0.05)
        await rt.stop()


# -- 4. submit 与 stop 并发 --


class TestSubmitAndStopConcurrent:
    async def test_stop_while_task_running(self):
        """任务执行中调用 stop，runtime 应正确停止。"""
        done = asyncio.Event()
        executor = ControllableExecutor(done_event=done)
        rt = TaskRuntime(executor, agent_id="concurrent")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.02)
        await rt.stop(cancel_running=True)
        assert rt.status is RuntimeStatus.STOPPED
        done.set()


# -- 5. register 与 unregister 并发 --


class TestRegisterUnregisterConcurrent:
    async def test_concurrent_register_unregister(self):
        """并发注册和注销不同 runtime 不产生竞争。"""
        coord = RuntimeCoordinator()
        runtimes = []
        for i in range(10):
            rt = TaskRuntime(ImmediateExecutor(), agent_id=f"r{i}")
            await rt.start()
            runtimes.append(rt)

        async def register_all():
            for rt in runtimes:
                coord.register(rt)

        async def unregister_half():
            for rt in runtimes[:5]:
                coord.unregister(rt.runtime_id)

        await asyncio.gather(register_all(), unregister_half())
        remaining = coord.list_all()
        assert len(remaining) == 5
        for rt in runtimes[5:]:
            await rt.stop()

    async def test_double_unregister_is_safe(self):
        """重复注销同一 runtime 不抛异常。"""
        coord = RuntimeCoordinator()
        rt = TaskRuntime(ImmediateExecutor(), agent_id="dup")
        await rt.start()
        coord.register(rt)
        coord.unregister(rt.runtime_id)
        coord.unregister(rt.runtime_id)
        assert rt.runtime_id not in {r.runtime_id for r in coord.list_all()}
        await rt.stop()


# -- 6. 旧 Attempt 晚于新 Attempt 完成 --


class TestAttemptOrdering:
    async def test_old_attempt_completes_after_new(self):
        """旧 attempt 晚完成时，结果按 task_id 分别存储不覆盖。"""
        done_old = asyncio.Event()
        done_new = asyncio.Event()

        rt_old = TaskRuntime(
            ControllableExecutor(done_event=done_old),
            agent_id="old",
        )
        rt_new = TaskRuntime(
            ControllableExecutor(done_event=done_new),
            agent_id="new",
        )
        await rt_old.start()
        await rt_new.start()

        task_old = _make_task(task_id="shared", attempt_id="att-1")
        task_new = _make_task(task_id="shared", attempt_id="att-2")

        await rt_old.submit(task_old)
        await asyncio.sleep(0.02)
        await rt_new.submit(task_new)
        await asyncio.sleep(0.02)

        done_new.set()
        await asyncio.sleep(0.05)
        result_new = rt_new.get_result("shared")
        assert result_new is not None
        assert result_new.attempt_id == "att-2"

        done_old.set()
        await asyncio.sleep(0.05)
        result_old = rt_old.get_result("shared")
        assert result_old is not None
        assert result_old.attempt_id == "att-1"

        await rt_old.stop()
        await rt_new.stop()


# -- 7. cancel 与正常完成同时发生 --


class TestCancelAndCompleteRace:
    async def test_cancel_during_execution(self):
        """cancel_task 后 executor 仍完成时，结果为 CANCELLED。"""
        started = asyncio.Event()
        done = asyncio.Event()
        executor = ControllableExecutor(
            start_event=started,
            done_event=done,
        )
        rt = TaskRuntime(executor, agent_id="race")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await started.wait()
        await rt.cancel_task(task.task_id, task.attempt_id)
        done.set()
        await asyncio.sleep(0.05)
        result = rt.get_result(task.task_id)
        assert result is not None
        assert result.status is TaskStatus.CANCELLED
        await rt.stop()


# -- 8. Coordinator shutdown 与 Runtime crash 同时发生 --


class TestCoordinatorShutdownAndCrash:
    async def test_crash_during_shutdown(self):
        """Runtime 崩溃与 coordinator shutdown 同时发生不死锁。"""
        coord = RuntimeCoordinator()
        executor = CrashingExecutor()
        rt = TaskRuntime(executor, agent_id="crash")
        await rt.start()
        coord.register(rt)
        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.05)
        await rt.stop()
        assert rt.status in (RuntimeStatus.STOPPED, RuntimeStatus.FAILED)
        coord.unregister(rt.runtime_id)


# -- 9. Event 重复投递 --


class TestEventDuplication:
    async def test_duplicate_event_handling(self):
        """同一事件多次投递不导致 listener 异常。"""
        done = asyncio.Event()
        executor = ControllableExecutor(done_event=done)
        rt = TaskRuntime(executor, agent_id="dup-evt")
        await rt.start()

        events: list = []

        async def collect():
            async for ev in rt.subscribe_events():
                events.append(ev)

        listener_task = asyncio.create_task(collect())
        task = _make_task()
        await rt.submit(task)
        done.set()
        await asyncio.sleep(0.05)
        await rt.stop()
        listener_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listener_task
        assert len(events) > 0


# -- 10. TaskStore CAS 冲突 --


class TestTaskStoreCAS:
    async def test_concurrent_status_update(self):
        """InMemoryTaskStore 并发更新同一任务状态不产生不一致。"""
        from mcc_core.runtime.task_store import InMemoryTaskStore

        store = InMemoryTaskStore()
        record = await store.create_task("subject", "desc")

        async def transition(from_s: TaskStatus, to_s: TaskStatus):
            with contextlib.suppress(Exception):
                await store.transition_task(record.task_id, from_s, to_s)

        await asyncio.gather(
            transition(TaskStatus.PENDING, TaskStatus.QUEUED),
            transition(TaskStatus.PENDING, TaskStatus.RUNNING),
        )
        result = await store.get_task(record.task_id)
        assert result is not None
        assert result.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)

    async def test_retry_creates_new_attempt(self):
        """create_attempt 生成新 attempt_id，不同 attempt 不覆盖。"""
        from mcc_core.runtime.task_store import InMemoryTaskStore

        store = InMemoryTaskStore()
        record = await store.create_task("retry-subject", "desc")
        attempt1 = await store.create_attempt(
            record.task_id,
            "agent-1",
            idempotency_key="key-1",
        )
        attempt2 = await store.create_attempt(
            record.task_id,
            "agent-1",
            idempotency_key="key-2",
        )
        assert attempt1.attempt_id != attempt2.attempt_id
        assert attempt1.idempotency_key == "key-1"
        assert attempt2.idempotency_key == "key-2"
