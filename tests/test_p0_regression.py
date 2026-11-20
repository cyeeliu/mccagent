"""P0 回归测试：验证四个关键缺陷修复。

P0-01: wait_for_result 不应只依赖事件流，应先查 TaskStore。
P0-02: cancel_task 应取消正在执行的 asyncio.Task。
P0-03: Runtime 应将完整结果（含 usage/cost）写入 TaskStore。
P0-04: wait_for_result 超时不应将 attempt 标记为 FAILED。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
from mcc_core import Usage
from mcc_core.runtime import (
    AttemptStatus,
    InMemoryTaskStore,
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeTask,
    TaskHandle,
    TaskStatus,
    TokenUsage,
)
from mcc_core.runtime.errors import TaskTimeoutError, TaskWaitTimeoutError
from mcc_core.runtime.result import RuntimeResult
from mcc_harness import RuntimeCoordinator, TaskDispatcher
from mcc_harness.agent import AgentRun

# -- 桩类 --


class _QuietRuntime:
    """Runtime 桩：接受 submit 但从不发事件，模拟事件丢失。"""

    def __init__(self, runtime_id: str = "quiet-rt") -> None:
        self._runtime_id = runtime_id
        self._event_listeners: list[asyncio.Queue[RuntimeEvent | None]] = []
        self._submitted_tasks: list[RuntimeTask] = []

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> Any:
        return None

    @property
    def alive(self) -> bool:
        return True

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_cancel=False,
            supports_health_check=False,
        )

    async def submit(self, task: RuntimeTask) -> TaskHandle:
        self._submitted_tasks.append(task)
        return TaskHandle(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
        )

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        return self._event_iterator(task_id=task_id, attempt_id=attempt_id)

    async def _event_iterator(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()
        self._event_listeners.append(queue)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                if task_id is not None and ev.task_id != task_id:
                    continue
                if attempt_id is not None and ev.attempt_id != attempt_id:
                    continue
                yield ev
        finally:
            if queue in self._event_listeners:
                self._event_listeners.remove(queue)

    async def stop(self, **kwargs: Any) -> None:
        for listener in self._event_listeners:
            await listener.put(None)


class _DelayedAgentLoop:
    """AgentLoop 桩：delay 后返回带 usage 的 AgentRun。"""

    def __init__(self, *, delay: float = 0.01) -> None:
        self._delay = delay

    async def run_streaming(
        self,
        prompt: str,
        queue: asyncio.Queue,
        *,
        instruction_queue: asyncio.Queue | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AgentRun:
        await asyncio.sleep(self._delay)
        await queue.put(None)
        return AgentRun(
            final_text=f"done: {prompt}",
            iterations=1,
            total_usage=Usage(input_tokens=20, output_tokens=10),
            cost_usd=0.001,
        )


class _StubExecutor:
    """AgentExecutor 桩：delay 后返回带 usage 的成功结果。"""

    def __init__(self, *, delay: float = 0.01) -> None:
        self._delay = delay
        self._closed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        await asyncio.sleep(self._delay)
        if cancellation is not None and cancellation.is_cancelled:
            return RuntimeResult(
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                runtime_id="stub",
                agent_id=task.agent_id,
                status=TaskStatus.CANCELLED,
                summary="cancelled",
            )
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="stub",
            agent_id=task.agent_id,
            status=TaskStatus.SUCCEEDED,
            summary=f"done: {task.objective}",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def close(self) -> None:
        self._closed = True


# -- 辅助函数 --


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


# -- P0-01: wait_for_result 不依赖事件流 --


class TestP0_01:
    async def test_wait_for_result_reads_store_when_no_event(self):
        """P0-01: 当结果已在 TaskStore 但事件丢失时，wait_for_result 应从 store 读取。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _QuietRuntime("quiet-rt")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="quiet-rt",
        )

        # 模拟 Runtime 已完成但事件丢失：直接写 TaskStore
        await store.complete_attempt(handle.attempt_id, summary="done quietly")

        # wait_for_result 应从 TaskStore 读到终态，返回合成事件
        event = await asyncio.wait_for(
            dispatcher.wait_for_result(handle, timeout=2.0),
            timeout=3.0,
        )
        assert event.event_type is RuntimeEventType.TASK_COMPLETED
        assert event.task_id == handle.task_id
        assert event.attempt_id == handle.attempt_id


# -- P0-02: cancel_task 取消运行中的 asyncio.Task --


class TestP0_02:
    async def test_cancel_task_cancels_running_task(self):
        """P0-02: cancel_task 应取消正在执行的 asyncio.Task。"""
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = _DelayedAgentLoop(delay=2.0)
        rt = InteractiveRuntimeV2(agent, agent_id="test-cancel")
        await rt.start()

        task = _make_task(objective="will be cancelled")
        await rt.submit(task)

        # submit 后应跟踪 _current_task
        assert rt._current_task is not None
        assert not rt._current_task.done()

        await rt.cancel_task(task.task_id, task.attempt_id, reason="test")

        # 等待取消传播
        await asyncio.sleep(0.1)
        assert rt._current_task is None or rt._current_task.done()

        await rt.stop()


# -- P0-03: Runtime 写完整结果到 TaskStore --


class TestP0_03:
    async def test_task_runtime_writes_full_result(self):
        """P0-03: TaskRuntime 应将完整结果（含 usage）写入 TaskStore。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        executor = _StubExecutor(delay=0.02)
        rt = TaskRuntime(executor, runtime_id="task-rt", task_store=store)
        await rt.start()
        coord.register(rt)

        dispatcher = TaskDispatcher(store, coord)
        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="task-rt",
        )

        event = await dispatcher.wait_for_result(handle, timeout=5.0)
        assert event.event_type is RuntimeEventType.TASK_COMPLETED

        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt.summary
        assert attempt.usage is not None
        assert attempt.usage.input_tokens > 0

        await rt.stop()

    async def test_interactive_runtime_writes_full_result(self):
        """P0-03: InteractiveRuntimeV2 应将完整结果（含 usage/cost）写入 TaskStore。"""
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        agent = _DelayedAgentLoop(delay=0.02)
        rt = InteractiveRuntimeV2(
            agent,
            agent_id="test-result",
            task_store=store,
        )
        await rt.start()
        coord.register(rt)

        dispatcher = TaskDispatcher(store, coord)
        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id=rt.runtime_id,
        )

        event = await dispatcher.wait_for_result(handle, timeout=5.0)
        assert event.event_type is RuntimeEventType.TASK_COMPLETED

        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt.summary
        assert attempt.usage is not None
        assert attempt.usage.input_tokens > 0

        await rt.stop()


# -- P0-04: 超时不标记 attempt 为 FAILED --


class TestP0_04:
    async def test_timeout_does_not_mark_attempt_failed(self):
        """P0-04: wait_for_result 超时不应将 attempt 标记为 FAILED。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _QuietRuntime("quiet-rt")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="quiet-rt",
        )

        with pytest.raises(TaskWaitTimeoutError):
            await dispatcher.wait_for_result(handle, timeout=0.05)

        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt.status is AttemptStatus.RUNNING
