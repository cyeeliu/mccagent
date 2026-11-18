"""InteractiveRuntimeV2：CLI 主 Agent 的交互式运行时（新 Protocol）。

多轮交互模型：start → ready → (submit → running → ready)* → stop → stopped。
通过 AgentLoopExecutor 执行任务，通过 subscribe_events 获取事件流。
保留 run_turn 方法供 CLI TUI 兼容使用。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from mcc_core import AgentEvent


class InteractiveRuntimeV2:
    """InteractiveRuntimeV2：实现 mcc_core.runtime.protocols.AgentRuntime。

    Phase 4 交付物：新 Protocol InteractiveRuntimeV2 实现。
    """

    def __init__(
        self,
        agent: Any,
        *,
        runtime_id: str | None = None,
        session_id: str = "",
        agent_id: str = "",
        task_store: Any | None = None,
    ) -> None:
        from mcc_core.runtime import INTERACTIVE_CAPABILITIES

        from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor

        self._agent = agent
        self._runtime_id = runtime_id or f"interactive::{uuid4().hex[:12]}"
        self._session_id = session_id
        self._agent_id = agent_id or self._runtime_id
        self._capabilities = INTERACTIVE_CAPABILITIES
        self._executor = AgentLoopExecutor(
            agent,
            runtime_id=self._runtime_id,
            agent_id=self._agent_id,
        )
        self._task_store = task_store

        from mcc_core.runtime import RuntimeStatus

        self._status = RuntimeStatus.CREATED
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._cancel_event: asyncio.Event | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._started_at: float = 0.0
        self._active_tasks: int = 0
        self._event_seq = 0
        self._event_listeners: list[asyncio.Queue] = []

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def status(self) -> Any:
        return self._status

    @property
    def alive(self) -> bool:
        from mcc_core.runtime import RuntimeStatus

        return self._status in (
            RuntimeStatus.STARTING,
            RuntimeStatus.READY,
            RuntimeStatus.PAUSING,
            RuntimeStatus.PAUSED,
        )

    @property
    def capabilities(self) -> Any:
        return self._capabilities

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def session_id(self) -> str:
        return self._session_id

    async def start(self) -> None:
        from mcc_core.runtime import RuntimeStatus

        if self._closed:
            raise RuntimeError(f"interactive runtime {self._runtime_id} is closed")
        if self._status is RuntimeStatus.READY:
            raise RuntimeError(f"interactive runtime {self._runtime_id} already started")
        self._started_at = time.monotonic()
        self._status = RuntimeStatus.STARTING
        await self._emit_event("runtime_started")
        self._status = RuntimeStatus.READY
        await self._emit_event("runtime_ready")

    async def submit(self, task: Any) -> Any:
        from mcc_core.runtime import RuntimeStatus
        from mcc_core.runtime.errors import RuntimeNotReadyError
        from mcc_core.runtime.handle import TaskHandle

        if self._status is not RuntimeStatus.READY:
            raise RuntimeNotReadyError(self._runtime_id, str(self._status))
        if self._active_tasks > 0:
            raise RuntimeNotReadyError(self._runtime_id, "已有活跃任务")
        self._active_tasks += 1
        handle = TaskHandle(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
        )
        await self._emit_event(
            "task_accepted",
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        )
        self._current_task = asyncio.create_task(self._execute_task(task))
        return handle

    async def _execute_task(self, task: Any) -> None:
        from mcc_core.runtime import RuntimeStatus

        self._status = RuntimeStatus.READY
        await self._emit_event(
            "task_started",
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        )
        try:
            result = await self._executor.execute(task)
            if self._task_store is not None:
                await self._write_result_to_store(task, result)
            event_type = "task_completed" if result.is_success else "task_failed"
            payload: dict[str, object] = {"status": str(result.status)}
            if result.summary:
                payload["summary"] = result.summary
            await self._emit_event(
                event_type,
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                payload=payload,
            )
        except Exception as exc:
            if self._task_store is not None:
                with contextlib.suppress(Exception):
                    await self._task_store.fail_attempt(
                        task.attempt_id, error=str(exc),
                    )
            await self._emit_event(
                "task_failed",
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                payload={"error": str(exc)},
            )
        finally:
            self._active_tasks -= 1
            self._current_task = None

    async def _write_result_to_store(self, task: Any, result: Any) -> None:
        """将 RuntimeResult 写入 TaskStore（P0-03）。"""
        if result.is_success:
            cost = float(result.cost_usd) if result.cost_usd is not None else None
            output = dict(result.output) if result.output else None
            await self._task_store.complete_attempt(
                task.attempt_id,
                summary=result.summary or None,
                output=output,
                usage=result.usage,
                cost_usd=cost,
            )
        else:
            error_msg = str(result.error) if result.error else (result.summary or "failed")
            await self._task_store.fail_attempt(
                task.attempt_id, error=error_msg,
            )

    async def cancel_task(
        self, task_id: str, attempt_id: str, reason: str | None = None,
    ) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
        await self._emit_event(
            "task_cancelled",
            task_id=task_id,
            attempt_id=attempt_id,
            payload={"reason": reason} if reason else {},
        )

    async def pause(self) -> None:
        from mcc_core.runtime.errors import UnsupportedOperationError

        raise UnsupportedOperationError("pause", self._runtime_id)

    async def resume(self) -> None:
        from mcc_core.runtime.errors import UnsupportedOperationError

        raise UnsupportedOperationError("resume", self._runtime_id)

    async def stop(
        self,
        timeout: float | None = None,
        *,
        cancel_running: bool = False,
    ) -> None:
        from mcc_core.runtime import RuntimeStatus

        if self._status in (RuntimeStatus.STOPPED, RuntimeStatus.FAILED):
            return
        self._closed = True
        self._status = RuntimeStatus.STOPPING
        await self._emit_event("runtime_stopping")
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                t = timeout if timeout is not None else 5.0
                await asyncio.wait_for(self._current_task, timeout=t)
        await self._executor.close()
        self._status = RuntimeStatus.STOPPED
        await self._emit_event("runtime_stopped")
        for listener in self._event_listeners:
            await listener.put(None)

    def cancel(self) -> None:
        """取消当前正在执行的 turn。"""
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    async def wait_done(self, *, timeout: float | None = None) -> Any:
        """等待当前任务完成。"""
        if self._current_task is not None:
            try:
                if timeout is not None:
                    await asyncio.wait_for(self._current_task, timeout=timeout)
                else:
                    await self._current_task
            except (TimeoutError, asyncio.CancelledError):
                pass
        return self._status, None

    async def health_check(self) -> Any:
        from mcc_core.runtime import HealthState, RuntimeHealth, RuntimeStatus

        uptime = time.monotonic() - self._started_at if self._started_at else 0.0
        state = HealthState.HEALTHY
        if self._status is RuntimeStatus.FAILED:
            state = HealthState.UNHEALTHY
        elif self._status in (RuntimeStatus.STOPPING, RuntimeStatus.STOPPED):
            state = HealthState.UNKNOWN
        return RuntimeHealth(
            runtime_id=self._runtime_id,
            state=state,
            active_tasks=self._active_tasks,
            uptime_seconds=uptime,
        )

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[Any]:
        return self._event_iterator(task_id=task_id, attempt_id=attempt_id)

    async def _event_iterator(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[Any]:
        queue: asyncio.Queue = asyncio.Queue()
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

    async def _emit_event(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        from datetime import UTC, datetime

        from mcc_core.runtime import RuntimeEvent, RuntimeEventType

        self._event_seq += 1
        event = RuntimeEvent(
            event_id=f"evt-{uuid4().hex[:12]}",
            event_type=RuntimeEventType(event_type),
            runtime_id=self._runtime_id,
            agent_id=self._agent_id,
            task_id=task_id,
            attempt_id=attempt_id,
            sequence=self._event_seq,
            occurred_at=datetime.now(UTC),
            payload=payload or {},
        )
        for listener in self._event_listeners:
            await listener.put(event)

    async def run_turn(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """CLI TUI 兼容接口：执行一个交互轮次，产出 AgentEvent 事件流。"""
        if self._closed:
            raise RuntimeError(
                f"interactive runtime {self._runtime_id} is closed"
            )
        async with self._turn_lock:
            self._cancel_event = asyncio.Event()
            queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

            async def worker() -> None:
                try:
                    await self._agent.run_streaming(
                        prompt,
                        queue,
                        cancel_event=self._cancel_event,
                    )
                except Exception as exc:
                    from mcc_core import Error as ErrorEvent

                    await queue.put(ErrorEvent(text=str(exc)))
                finally:
                    await queue.put(None)

            task = asyncio.create_task(worker())
            self._current_task = task
            try:
                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    yield ev
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                self._current_task = None
                self._cancel_event = None
