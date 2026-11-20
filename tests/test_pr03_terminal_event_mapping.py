"""PR-03: 终态事件映射闭环（P0-03）。

验证：
- executor 返回 CANCELLED 时，PersistentRuntimeV2 转发 TASK_CANCELLED 事件
- executor 返回 TIMED_OUT 时，PersistentRuntimeV2 转发 TASK_TIMED_OUT 事件
- 事件 payload 携带完整结果字段
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


class TerminalResultExecutor:
    """返回指定终态结果的桩执行器。"""

    def __init__(self, status: TaskStatus, summary: str = "") -> None:
        self._status = status
        self._summary = summary
        self._closed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="stub",
            agent_id=task.agent_id,
            status=self._status,
            summary=self._summary or self._status.value,
        )

    async def close(self) -> None:
        self._closed = True


def _make_task() -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="terminal event",
    )


async def _collect_event(rt: Any, target: RuntimeEventType) -> list[Any]:
    """收集直到目标事件出现或 runtime 停止。"""
    collected: list[Any] = []

    async def collect() -> None:
        async for ev in rt.subscribe_events():
            collected.append(ev)
            if ev.event_type in (target, RuntimeEventType.RUNTIME_STOPPED):
                break

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0.01)
    return collected, collector


class TestPR03TerminalEventMapping:
    """PR-03: PersistentRuntimeV2 终态事件映射闭环。"""

    async def test_cancelled_event_mapped(self) -> None:
        """executor 返回 CANCELLED 时转发 TASK_CANCELLED 事件。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = TerminalResultExecutor(TaskStatus.CANCELLED, "cancelled")
        rt = PersistentRuntimeV2(executor, agent_id="pr03-cancel-test")
        await rt.start()
        collected, collector = await _collect_event(
            rt, RuntimeEventType.TASK_CANCELLED
        )

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.1)

        types = [e.event_type for e in collected]
        assert RuntimeEventType.TASK_CANCELLED in types, (
            f"TASK_CANCELLED 事件被丢弃，收到: {types}"
        )
        await rt.stop()
        if not collector.done():
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector

    async def test_timed_out_event_mapped(self) -> None:
        """executor 返回 TIMED_OUT 时转发 TASK_TIMED_OUT 事件。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = TerminalResultExecutor(TaskStatus.TIMED_OUT, "timed out")
        rt = PersistentRuntimeV2(executor, agent_id="pr03-timeout-test")
        await rt.start()
        collected, collector = await _collect_event(
            rt, RuntimeEventType.TASK_TIMED_OUT
        )

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.1)

        types = [e.event_type for e in collected]
        assert RuntimeEventType.TASK_TIMED_OUT in types, (
            f"TASK_TIMED_OUT 事件被丢弃，收到: {types}"
        )
        await rt.stop()
        if not collector.done():
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector

    async def test_cancelled_payload_carries_status(self) -> None:
        """TASK_CANCELLED 事件 payload 携带 status 字段。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        executor = TerminalResultExecutor(TaskStatus.CANCELLED, "cancelled")
        rt = PersistentRuntimeV2(executor, agent_id="pr03-payload-test")
        await rt.start()
        collected, collector = await _collect_event(
            rt, RuntimeEventType.TASK_CANCELLED
        )

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.1)

        cancel_events = [
            e for e in collected if e.event_type is RuntimeEventType.TASK_CANCELLED
        ]
        assert cancel_events, "未收到 TASK_CANCELLED 事件"
        assert "status" in cancel_events[0].payload, "payload 缺少 status 字段"
        await rt.stop()
        if not collector.done():
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector
