"""TaskRuntime：短生命周期运行时，替代 EphemeralRuntime。

单次执行模型：start → ready → submit → running → terminal → stop。
通过 AgentExecutor 执行任务，不直接依赖 SubagentRunner。
不支持 pause/resume（短生命周期无暂停需求）。

Phase 4 交付物：新 Protocol TaskRuntime，替代 EphemeralRuntime。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from mcc_core.runtime import (
    HealthState,
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeStatus,
    TaskHandle,
)
from mcc_core.runtime.errors import (
    RuntimeNotReadyError,
    UnsupportedOperationError,
)
from mcc_core.runtime.handle import TaskHandle as TaskHandleType
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class TaskRuntime:
    """短生命周期运行时：通过 AgentExecutor 执行单个任务。

    生命周期：CREATED → STARTING → READY → (submit → RUNNING) → STOPPED。
    单任务执行，不支持并发提交多个任务。
    cancel_task 通过 CancellationToken 协作取消。
    """

    def __init__(
        self,
        executor: Any,
        *,
        runtime_id: str | None = None,
        agent_id: str = "",
        capabilities: RuntimeCapabilities | None = None,
        task_store: Any | None = None,
    ) -> None:
        from mcc_core.runtime import TASK_CAPABILITIES

        self._executor = executor
        self._runtime_id = runtime_id or f"task::{uuid4().hex[:12]}"
        self._agent_id = agent_id or self._runtime_id
        self._capabilities = capabilities or TASK_CAPABILITIES
        self._task_store = task_store
        self._status = RuntimeStatus.CREATED
        self._started_at: float = 0.0
        self._active_tasks: int = 0
        self._event_seq = 0
        self._event_listeners: list[asyncio.Queue[RuntimeEvent | None]] = []
        self._cancel_tokens: dict[str, Any] = {}
        self._results: dict[str, RuntimeResult] = {}
        self._lock = asyncio.Lock()

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status in (RuntimeStatus.READY, RuntimeStatus.STARTING)

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    async def start(self) -> None:
        if self._status is RuntimeStatus.READY:
            raise RuntimeError(f"task runtime {self._runtime_id} already started")
        self._started_at = time.monotonic()
        self._status = RuntimeStatus.STARTING
        await self._emit(RuntimeEventType.RUNTIME_STARTED)
        self._status = RuntimeStatus.READY
        await self._emit(RuntimeEventType.RUNTIME_READY)

    async def submit(self, task: RuntimeTask) -> TaskHandleType:
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
        await self._emit(
            RuntimeEventType.TASK_ACCEPTED,
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        )
        asyncio.create_task(self._execute(task))
        return handle

    async def _execute(self, task: RuntimeTask) -> None:
        from mcc_harness.execution.cancellation import CancellationToken

        cancel_token = CancellationToken()
        self._cancel_tokens[task.task_id] = cancel_token

        await self._emit(
            RuntimeEventType.TASK_STARTED,
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        )

        try:
            result = await self._executor.execute(
                task,
                cancellation=cancel_token,
            )
            self._results[task.task_id] = result
            if self._task_store is not None:
                await self._write_result_to_store(task, result)
            event_type = (
                RuntimeEventType.TASK_COMPLETED
                if result.is_success
                else RuntimeEventType.TASK_FAILED
            )
            payload: dict[str, object] = {"status": str(result.status)}
            if result.summary:
                payload["summary"] = result.summary
            await self._emit(
                event_type,
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                payload=payload,
            )
        except asyncio.CancelledError:
            if self._task_store is not None:
                with contextlib.suppress(Exception):
                    await self._task_store.fail_attempt(
                        task.attempt_id, error="cancelled",
                    )
            await self._emit(
                RuntimeEventType.TASK_CANCELLED,
                task_id=task.task_id,
                attempt_id=task.attempt_id,
            )
        except Exception as exc:
            if self._task_store is not None:
                with contextlib.suppress(Exception):
                    await self._task_store.fail_attempt(
                        task.attempt_id, error=str(exc),
                    )
            await self._emit(
                RuntimeEventType.TASK_FAILED,
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                payload={"error": str(exc)},
            )
        finally:
            self._cancel_tokens.pop(task.task_id, None)
            self._active_tasks -= 1

    async def _write_result_to_store(self, task: RuntimeTask, result: RuntimeResult) -> None:
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
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None:
        token = self._cancel_tokens.get(task_id)
        if token is not None:
            token.cancel(reason)
        await self._emit(
            RuntimeEventType.TASK_CANCELLED,
            task_id=task_id,
            attempt_id=attempt_id,
            payload={"reason": reason} if reason else {},
        )

    async def pause(self) -> None:
        raise UnsupportedOperationError("pause", self._runtime_id)

    async def resume(self) -> None:
        raise UnsupportedOperationError("resume", self._runtime_id)

    async def stop(
        self,
        timeout: float | None = None,
        *,
        cancel_running: bool = False,
    ) -> None:
        if self._status in (RuntimeStatus.STOPPED, RuntimeStatus.FAILED):
            return
        self._status = RuntimeStatus.STOPPING
        await self._emit(RuntimeEventType.RUNTIME_STOPPING)
        if cancel_running:
            for token in self._cancel_tokens.values():
                token.cancel("runtime stopping")
        if hasattr(self._executor, "close"):
            await self._executor.close()
        self._status = RuntimeStatus.STOPPED
        await self._emit(RuntimeEventType.RUNTIME_STOPPED)
        for listener in self._event_listeners:
            await listener.put(None)

    async def health_check(self) -> RuntimeHealth:
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

    async def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self._event_seq += 1
        event = RuntimeEvent(
            event_id=f"evt-{uuid4().hex[:12]}",
            event_type=event_type,
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

    def get_result(self, task_id: str) -> RuntimeResult | None:
        return self._results.get(task_id)


__all__ = ["TaskRuntime"]
