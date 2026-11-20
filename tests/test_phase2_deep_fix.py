"""Phase 2 深度修复验证：结果契约闭环 + 事件竞态消除。

P0-03: TaskRuntime / InteractiveRuntimeV2 事件 payload 携带完整结果：
       - result_ref 指向持久化结果
       - output / usage / cost_usd / artifacts 不丢失
       - Dispatcher._record_result 从 payload 读取完整数据

P0-01: wait_for_result 先订阅再查询，消除竞态窗口。
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import InMemoryTaskStore, TaskStatus, TokenUsage
from mcc_core.runtime.task import RuntimeTask


class SlowStubAgentLoop:
    """慢速桩 AgentLoop：支持取消，用于测试事件 payload。"""

    def __init__(self, *, delay: float = 0.5) -> None:
        self._delay = delay

    async def run_streaming(
        self,
        prompt: str,
        queue: asyncio.Queue,
        *,
        instruction_queue: asyncio.Queue | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> Any:
        from mcc_core import Usage
        from mcc_harness.agent import AgentRun

        try:
            if cancel_event is not None:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(asyncio.sleep(self._delay)),
                        asyncio.create_task(cancel_event.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                if cancel_event.is_set():
                    await queue.put(None)
                    return AgentRun(
                        final_text="",
                        iterations=0,
                        total_usage=Usage(),
                        cost_usd=0.0,
                    )
            else:
                await asyncio.sleep(self._delay)
            await queue.put(None)
            return AgentRun(
                final_text=f"done: {prompt}",
                iterations=1,
                total_usage=Usage(),
                cost_usd=0.0,
            )
        except asyncio.CancelledError:
            await queue.put(None)
            return AgentRun(
                final_text="",
                iterations=0,
                total_usage=Usage(),
                cost_usd=0.0,
            )


class StubExecutor:
    """桩 AgentExecutor：返回带完整字段的 RuntimeResult。"""

    def __init__(self, *, delay: float = 0.01) -> None:
        self._delay = delay
        self._closed = False

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
        from mcc_core.runtime import RuntimeResult

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


async def _collect_done_event(rt: Any, task: RuntimeTask) -> Any:
    """并发提交任务并收集完成事件。"""
    done_event: Any = None

    async def _wait() -> None:
        nonlocal done_event
        async for ev in rt.subscribe_events(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        ):
            if ev.event_type.value in (
                "task_completed",
                "task_failed",
                "task_cancelled",
            ):
                done_event = ev
                return

    waiter = asyncio.create_task(_wait())
    await rt.submit(task)
    await asyncio.sleep(0.15)
    if done_event is None:
        await asyncio.wait_for(waiter, timeout=5.0)
    else:
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await waiter
    return done_event


# -- P0-03: 事件 payload 完整性 --


class TestEventPayloadCompleteness:
    async def test_task_runtime_event_includes_result_ref(self):
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = StubExecutor()
        rt = TaskRuntime(executor, agent_id="task-test")
        await rt.start()
        task = _make_task()
        done_event = await _collect_done_event(rt, task)
        await rt.stop()
        assert done_event is not None
        payload = done_event.payload
        assert "result_ref" in payload, "事件 payload 应包含 result_ref"

    async def test_task_runtime_event_includes_full_result(self):
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = StubExecutor()
        rt = TaskRuntime(executor, agent_id="task-test")
        await rt.start()
        task = _make_task()
        done_event = await _collect_done_event(rt, task)
        await rt.stop()
        assert done_event is not None
        payload = done_event.payload
        assert payload.get("output") == {"answer": 42}, "事件 payload 应包含 output"
        assert "usage" in payload, "事件 payload 应包含 usage"
        assert payload.get("cost_usd") is not None, "事件 payload 应包含 cost_usd"

    async def test_interactive_event_includes_full_result(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=0.05)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        task = _make_task()
        done_event = await _collect_done_event(rt, task)
        await rt.stop(timeout=5.0)
        assert done_event is not None
        payload = done_event.payload
        assert "result_ref" in payload, "事件 payload 应包含 result_ref"

    async def test_dispatcher_record_result_from_payload(self):
        from mcc_harness.runtime.coordinator import RuntimeCoordinator
        from mcc_harness.runtime.dispatcher import TaskDispatcher
        from mcc_harness.runtime.task_runtime import TaskRuntime

        store = InMemoryTaskStore()
        coordinator = RuntimeCoordinator()
        executor = StubExecutor()
        rt = TaskRuntime(executor, agent_id="task-test", task_store=store)
        await rt.start()
        coordinator.register(rt)
        dispatcher = TaskDispatcher(store, coordinator)

        record = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            record.task_id,
            "task-test",
            runtime_id=rt.runtime_id,
        )
        event = await dispatcher.wait_for_result(handle, timeout=5.0)
        assert event.event_type.value == "task_completed"

        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt is not None
        assert attempt.summary is not None
        await rt.stop()


# -- P0-01: wait_for_result 先订阅再查询消除竞态 --


class TestWaitForResultRaceFree:
    """验证 wait_for_result 先订阅事件再查询状态，不丢事件。"""

    async def test_wait_after_completion_returns_synthetic_event(self):
        """任务已完成后调用 wait_for_result 仍能返回结果（合成事件）。"""
        from mcc_harness.runtime.coordinator import RuntimeCoordinator
        from mcc_harness.runtime.dispatcher import TaskDispatcher
        from mcc_harness.runtime.task_runtime import TaskRuntime

        store = InMemoryTaskStore()
        coordinator = RuntimeCoordinator()
        executor = StubExecutor()
        rt = TaskRuntime(executor, agent_id="task-test", task_store=store)
        await rt.start()
        coordinator.register(rt)
        dispatcher = TaskDispatcher(store, coordinator)

        record = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            record.task_id,
            "task-test",
            runtime_id=rt.runtime_id,
        )
        # 先等待完成，确保 attempt 已终态
        first = await dispatcher.wait_for_result(handle, timeout=5.0)
        assert first.event_type.value == "task_completed"
        # 再次调用：attempt 已终态，应直接返回合成事件而不阻塞
        second = await dispatcher.wait_for_result(handle, timeout=1.0)
        assert second.event_type.value == "task_completed"
        await rt.stop()
