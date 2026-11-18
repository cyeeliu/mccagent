"""新 Protocol Runtime 实现的测试。

验证 InteractiveRuntimeV2、TaskRuntime、PersistentRuntimeV2 满足
AgentRuntime Protocol 契约（Contract Tests）以及各自特有行为。

Phase 4 交付物：新 Protocol Runtime 测试。
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import (
    INTERACTIVE_CAPABILITIES,
    PERSISTENT_CAPABILITIES,
    TASK_CAPABILITIES,
    RuntimeStatus,
    TaskStatus,
)
from mcc_core.runtime.task import RuntimeTask


class StubAgentLoop:
    """桩 AgentLoop：不依赖 LLM，满足 AgentLoopExecutor 调用需求。"""

    def __init__(self) -> None:
        self._closed = False

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

        await queue.put(None)
        return AgentRun(
            final_text=f"done: {prompt}",
            iterations=1,
            total_usage=Usage(),
            cost_usd=0.0,
        )


class StubExecutor:
    """桩 AgentExecutor：不依赖 AgentLoop，直接返回成功结果。"""

    def __init__(self, *, delay: float = 0.01) -> None:
        self._delay = delay
        self._closed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> Any:
        from mcc_core.runtime import RuntimeResult, TokenUsage

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


@pytest.fixture
def make_task() -> Any:
    return _make_task


@pytest.fixture
async def interactive_runtime_factory() -> Any:
    from mcc_harness.runtime.interactive import InteractiveRuntimeV2

    async def _factory() -> Any:
        agent = StubAgentLoop()
        return InteractiveRuntimeV2(agent, agent_id="interactive-test")

    return _factory


@pytest.fixture
async def task_runtime_factory() -> Any:
    from mcc_harness.runtime.task_runtime import TaskRuntime

    async def _factory() -> Any:
        executor = StubExecutor()
        return TaskRuntime(executor, agent_id="task-test")

    return _factory


class TestInteractiveRuntimeV2Contract:
    """InteractiveRuntimeV2 契约测试。"""

    async def test_initial_status_created(
        self, interactive_runtime_factory: Any,
    ) -> None:
        rt = await interactive_runtime_factory()
        assert rt.status is RuntimeStatus.CREATED

    async def test_start_transitions_to_ready(
        self, interactive_runtime_factory: Any,
    ) -> None:
        rt = await interactive_runtime_factory()
        await rt.start()
        assert rt.status is RuntimeStatus.READY
        await rt.stop()

    async def test_start_twice_raises(
        self, interactive_runtime_factory: Any,
    ) -> None:
        rt = await interactive_runtime_factory()
        await rt.start()
        with pytest.raises(RuntimeError):
            await rt.start()
        await rt.stop()

    async def test_submit_before_start_raises(
        self, interactive_runtime_factory: Any, make_task: Any,
    ) -> None:
        from mcc_core.runtime.errors import RuntimeNotReadyError

        rt = await interactive_runtime_factory()
        task = make_task()
        with pytest.raises(RuntimeNotReadyError):
            await rt.submit(task)

    async def test_submit_returns_handle(
        self, interactive_runtime_factory: Any, make_task: Any,
    ) -> None:
        from mcc_core.runtime import TaskHandle

        rt = await interactive_runtime_factory()
        await rt.start()
        task = make_task()
        handle = await rt.submit(task)
        assert isinstance(handle, TaskHandle)
        assert handle.task_id == task.task_id
        assert handle.runtime_id == rt.runtime_id
        await asyncio.sleep(0.05)
        await rt.stop()

    async def test_stop_transitions_to_stopped(
        self, interactive_runtime_factory: Any,
    ) -> None:
        rt = await interactive_runtime_factory()
        await rt.start()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_stop_is_idempotent(
        self, interactive_runtime_factory: Any,
    ) -> None:
        rt = await interactive_runtime_factory()
        await rt.start()
        await rt.stop()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_health_check(
        self, interactive_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime import RuntimeHealth

        rt = await interactive_runtime_factory()
        await rt.start()
        health = await rt.health_check()
        assert isinstance(health, RuntimeHealth)
        assert health.runtime_id == rt.runtime_id
        await rt.stop()

    async def test_pause_unsupported(
        self, interactive_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime.errors import UnsupportedOperationError

        rt = await interactive_runtime_factory()
        await rt.start()
        with pytest.raises(UnsupportedOperationError):
            await rt.pause()
        await rt.stop()

    async def test_capabilities(self, interactive_runtime_factory: Any) -> None:
        rt = await interactive_runtime_factory()
        assert rt.capabilities is INTERACTIVE_CAPABILITIES
        assert not rt.capabilities.supports_pause
        assert not rt.capabilities.supports_multiple_tasks

    async def test_events_emitted(
        self, interactive_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime import RuntimeEvent

        rt = await interactive_runtime_factory()
        collected: list[RuntimeEvent] = []

        async def collect() -> None:
            async for ev in rt.subscribe_events():
                collected.append(ev)
                if ev.event_type.value == "runtime_stopped":
                    break

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        await rt.start()
        await asyncio.sleep(0.01)
        await rt.stop()
        await asyncio.sleep(0.05)
        if not collector.done():
            collector.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await collector

        types = [e.event_type.value for e in collected]
        assert "runtime_started" in types
        assert "runtime_ready" in types


class TestTaskRuntimeContract:
    """TaskRuntime 契约测试。"""

    async def test_initial_status_created(
        self, task_runtime_factory: Any,
    ) -> None:
        rt = await task_runtime_factory()
        assert rt.status is RuntimeStatus.CREATED

    async def test_start_transitions_to_ready(
        self, task_runtime_factory: Any,
    ) -> None:
        rt = await task_runtime_factory()
        await rt.start()
        assert rt.status is RuntimeStatus.READY
        await rt.stop()

    async def test_submit_returns_handle(
        self, task_runtime_factory: Any, make_task: Any,
    ) -> None:
        from mcc_core.runtime import TaskHandle

        rt = await task_runtime_factory()
        await rt.start()
        task = make_task()
        handle = await rt.submit(task)
        assert isinstance(handle, TaskHandle)
        assert handle.task_id == task.task_id
        await asyncio.sleep(0.05)
        await rt.stop()

    async def test_submit_executes_task(
        self, task_runtime_factory: Any, make_task: Any,
    ) -> None:
        rt = await task_runtime_factory()
        await rt.start()
        task = make_task(objective="hello world")
        await rt.submit(task)
        await asyncio.sleep(0.05)
        result = rt.get_result(task.task_id)
        assert result is not None
        assert result.is_success
        assert "hello world" in result.summary
        await rt.stop()

    async def test_cancel_task(
        self, task_runtime_factory: Any, make_task: Any,
    ) -> None:
        from mcc_harness.execution.cancellation import CancellationToken
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = StubExecutor(delay=1.0)
        rt = TaskRuntime(executor, agent_id="task-cancel-test")
        await rt.start()
        task = make_task()
        await rt.submit(task)
        await asyncio.sleep(0.01)
        await rt.cancel_task(task.task_id, task.attempt_id, "test cancel")
        await asyncio.sleep(0.05)
        await rt.stop()

    async def test_stop_transitions_to_stopped(
        self, task_runtime_factory: Any,
    ) -> None:
        rt = await task_runtime_factory()
        await rt.start()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_pause_unsupported(
        self, task_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime.errors import UnsupportedOperationError

        rt = await task_runtime_factory()
        await rt.start()
        with pytest.raises(UnsupportedOperationError):
            await rt.pause()
        await rt.stop()

    async def test_capabilities(self, task_runtime_factory: Any) -> None:
        rt = await task_runtime_factory()
        assert rt.capabilities is TASK_CAPABILITIES
        assert not rt.capabilities.supports_pause
        assert not rt.capabilities.supports_multiple_tasks

    async def test_health_check(
        self, task_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime import RuntimeHealth

        rt = await task_runtime_factory()
        await rt.start()
        health = await rt.health_check()
        assert isinstance(health, RuntimeHealth)
        assert health.runtime_id == rt.runtime_id
        await rt.stop()

    async def test_events_on_lifecycle(
        self, task_runtime_factory: Any,
    ) -> None:
        from mcc_core.runtime import RuntimeEvent

        rt = await task_runtime_factory()
        collected: list[RuntimeEvent] = []

        async def collect() -> None:
            async for ev in rt.subscribe_events():
                collected.append(ev)
                if ev.event_type.value == "runtime_stopped":
                    break

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        await rt.start()
        await asyncio.sleep(0.01)
        await rt.stop()
        await asyncio.sleep(0.05)
        if not collector.done():
            collector.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await collector

        types = [e.event_type.value for e in collected]
        assert "runtime_started" in types
        assert "runtime_ready" in types


class TestPersistentRuntimeV2:
    """PersistentRuntimeV2 测试（不使用完整 Contract Tests，因依赖 TeammateRuntime）。"""

    async def test_capabilities(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class StubTeammate:
            spec = type("S", (), {"name": "test"})()
            def runtime_status(self) -> Any: ...
            def notify(self) -> None: ...
            def alive(self) -> bool: ...
            async def start(self, *a: Any, **k: Any) -> None: ...
            async def shutdown(self, *a: Any, **k: Any) -> None: ...
            def set_task_future(self, *a: Any) -> None: ...

        class StubBus:
            async def send(self, *a: Any, **k: Any) -> None: ...

        teammate = StubTeammate()
        rt = PersistentRuntimeV2.from_teammate(teammate, StubBus())
        assert rt.capabilities is PERSISTENT_CAPABILITIES
        assert rt.capabilities.supports_pause
        assert rt.capabilities.supports_resume
        assert rt.capabilities.supports_multiple_tasks

    async def test_status_mapping(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class StubTeammate:
            spec = type("S", (), {"name": "test"})()
            runtime_status = RuntimeStatus.CREATED
            def notify(self) -> None: ...
            def alive(self) -> bool: return False
            async def start(self, *a: Any, **k: Any) -> None: ...
            async def shutdown(self, *a: Any, **k: Any) -> None: ...
            def set_task_future(self, *a: Any) -> None: ...

        class StubBus:
            async def send(self, *a: Any, **k: Any) -> None: ...

        teammate = StubTeammate()
        rt = PersistentRuntimeV2.from_teammate(teammate, StubBus())
        assert rt.status is RuntimeStatus.CREATED


class TestBaseRuntime:
    """BaseRuntime 基类单元测试。"""

    async def test_state_machine(self) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES
        from mcc_harness.runtime.base_runtime import BaseRuntime

        rt = BaseRuntime(
            agent_id="test",
            capabilities=INTERACTIVE_CAPABILITIES,
        )
        assert rt.status is RuntimeStatus.CREATED
        await rt.start()
        assert rt.status is RuntimeStatus.READY
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_invalid_transition(self) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES
        from mcc_core.runtime.errors import InvalidTransitionError
        from mcc_harness.runtime.base_runtime import BaseRuntime

        rt = BaseRuntime(
            agent_id="test",
            capabilities=INTERACTIVE_CAPABILITIES,
        )
        with pytest.raises(InvalidTransitionError):
            rt._transition(RuntimeStatus.READY)

    async def test_health_check(self) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES, HealthState
        from mcc_harness.runtime.base_runtime import BaseRuntime

        rt = BaseRuntime(
            agent_id="test",
            capabilities=INTERACTIVE_CAPABILITIES,
        )
        await rt.start()
        health = await rt.health_check()
        assert health.is_healthy
        await rt.stop()
        health = await rt.health_check()
        assert health.state is HealthState.UNKNOWN

    async def test_subscribe_events(self) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES, RuntimeEvent
        from mcc_harness.runtime.base_runtime import BaseRuntime

        rt = BaseRuntime(
            agent_id="test",
            capabilities=INTERACTIVE_CAPABILITIES,
        )
        collected: list[RuntimeEvent] = []

        async def collect() -> None:
            async for ev in rt.subscribe_events():
                collected.append(ev)
                if ev.event_type.value == "runtime_stopped":
                    break

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.01)
        await rt.start()
        await asyncio.sleep(0.01)
        await rt.stop()
        await asyncio.sleep(0.05)
        if not collector.done():
            collector.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError):
                await collector

        types = [e.event_type.value for e in collected]
        assert "runtime_started" in types
        assert "runtime_ready" in types
        assert "runtime_stopping" in types
        assert "runtime_stopped" in types

    async def test_submit_before_start(self) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES
        from mcc_core.runtime.errors import RuntimeNotReadyError
        from mcc_harness.runtime.base_runtime import BaseRuntime

        rt = BaseRuntime(
            agent_id="test",
            capabilities=INTERACTIVE_CAPABILITIES,
        )
        task = _make_task()
        with pytest.raises(RuntimeNotReadyError):
            await rt.submit(task)
