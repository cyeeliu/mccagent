"""统一 Runtime Contract Tests。

验证所有新 Protocol Runtime 实现满足 AgentRuntime Protocol 的行为契约。
具体 Runtime 测试文件通过 override `runtime_factory` fixture 复用此套件。

Phase 4 交付物：统一行为契约测试（G-12）。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import pytest
from mcc_core.runtime import (
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeHealth,
    RuntimeStatus,
    TaskHandle,
)
from mcc_core.runtime.errors import (
    InvalidTransitionError,
    RuntimeNotReadyError,
    UnsupportedOperationError,
)
from mcc_core.runtime.task import RuntimeTask

RuntimeFactory = Callable[[], Coroutine[Any, Any, Any]]


@pytest.fixture
def make_task() -> Callable[..., RuntimeTask]:
    from uuid import uuid4

    def _make(
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

    return _make


class TestRuntimeContract:
    """所有 Runtime 实现必须通过的契约测试。

    此类为基类，不直接运行。具体 Runtime 测试文件继承此类并
    override runtime_factory fixture 后由 pytest 收集。
    """

    __test__ = False

    async def test_initial_status_created(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        assert rt.status is RuntimeStatus.CREATED

    async def test_runtime_id_is_string(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        assert isinstance(rt.runtime_id, str)
        assert len(rt.runtime_id) > 0

    async def test_agent_id_is_string(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        assert isinstance(rt.agent_id, str)

    async def test_capabilities_is_runtime_capabilities(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        assert isinstance(rt.capabilities, RuntimeCapabilities)

    async def test_start_transitions_to_ready(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        assert rt.status is RuntimeStatus.READY
        await rt.stop()

    async def test_start_twice_raises_invalid_transition(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        with pytest.raises(InvalidTransitionError):
            await rt.start()
        await rt.stop()

    async def test_submit_before_start_raises_not_ready(
        self, runtime_factory: RuntimeFactory, make_task: Callable[..., RuntimeTask],
    ) -> None:
        rt = await runtime_factory()
        task = make_task()
        with pytest.raises(RuntimeNotReadyError):
            await rt.submit(task)

    async def test_submit_returns_task_handle(
        self, runtime_factory: RuntimeFactory, make_task: Callable[..., RuntimeTask],
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        task = make_task()
        handle = await rt.submit(task)
        assert isinstance(handle, TaskHandle)
        assert handle.task_id == task.task_id
        assert handle.attempt_id == task.attempt_id
        assert handle.runtime_id == rt.runtime_id
        await rt.stop()

    async def test_stop_transitions_to_stopped(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_stop_is_idempotent(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        await rt.stop()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_stop_before_start_is_safe(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.stop()
        assert rt.status is RuntimeStatus.STOPPED

    async def test_health_check_returns_health(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        health = await rt.health_check()
        assert isinstance(health, RuntimeHealth)
        assert health.runtime_id == rt.runtime_id
        await rt.stop()

    async def test_subscribe_events_returns_async_iterator(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        events = rt.subscribe_events()
        assert hasattr(events, "__aiter__")
        await rt.start()
        await rt.stop()

    async def test_pause_unsupported_raises(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        if not rt.capabilities.supports_pause:
            with pytest.raises(UnsupportedOperationError):
                await rt.pause()
        await rt.stop()

    async def test_resume_unsupported_raises(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        if not rt.capabilities.supports_resume:
            with pytest.raises(UnsupportedOperationError):
                await rt.resume()
        await rt.stop()

    async def test_cancel_task_unsupported_raises(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
        await rt.start()
        if not rt.capabilities.supports_cancel_task:
            with pytest.raises(UnsupportedOperationError):
                await rt.cancel_task("t1", "a1")
        await rt.stop()

    async def test_events_emitted_on_lifecycle(
        self, runtime_factory: RuntimeFactory,
    ) -> None:
        rt = await runtime_factory()
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
            with contextlib_suppress():
                await collector

        event_types = [e.event_type.value for e in collected]
        assert "runtime_started" in event_types
        assert "runtime_ready" in event_types
        assert "runtime_stopping" in event_types or "runtime_stopped" in event_types


def contextlib_suppress() -> Any:
    import contextlib

    return contextlib.suppress(asyncio.CancelledError, TimeoutError)
