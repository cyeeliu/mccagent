"""PersistentRuntimeV2：常驻运行时（新 Protocol），原生 BoundedTaskQueue + WorkerPool。

循环执行模型：start → ready → (submit → running → ready)* → stop → stopped。
通过 BoundedTaskQueue + WorkerPool + AgentExecutor 执行任务，
支持背压控制和并发上限。
通过 from_teammate() classmethod 提供 TeammateRuntime 向后兼容。
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from mcc_core.runtime import (
    PERSISTENT_CAPABILITIES,
    HealthState,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    TaskHandle,
    TaskStatus,
)
from mcc_core.runtime import (
    RuntimeStatus as CoreRuntimeStatus,
)
from mcc_core.runtime.result import RuntimeResult as CoreRuntimeResult
from mcc_core.runtime.task import RuntimeTask

from .base import (
    RuntimeResult,
    RuntimeStatus,
    RuntimeStopTimeoutError,
    WaitTimeoutError,
    _terminal_result,
)
from .bounded_task_queue import BoundedTaskQueue
from .worker_pool import WorkerPool

_OLD_TO_NEW_STATUS: dict[RuntimeStatus, str] = {
    RuntimeStatus.CREATED: "created",
    RuntimeStatus.STARTING: "starting",
    RuntimeStatus.RUNNING: "ready",
    RuntimeStatus.IDLE: "ready",
    RuntimeStatus.PAUSED: "paused",
    RuntimeStatus.STOPPING: "stopping",
    RuntimeStatus.COMPLETED: "ready",
    RuntimeStatus.FAILED: "failed",
    RuntimeStatus.CANCELLED: "ready",
    RuntimeStatus.TIMEOUT: "ready",
    RuntimeStatus.BUDGET_EXCEEDED: "ready",
    RuntimeStatus.SHUTDOWN: "stopped",
}

_TASK_STATUS_TO_RUNTIME: dict[TaskStatus, RuntimeStatus] = {
    TaskStatus.SUCCEEDED: RuntimeStatus.COMPLETED,
    TaskStatus.FAILED: RuntimeStatus.FAILED,
    TaskStatus.CANCELLED: RuntimeStatus.CANCELLED,
    TaskStatus.TIMED_OUT: RuntimeStatus.TIMEOUT,
}

_TASK_OUTCOME_TO_RUNTIME: dict[Any, RuntimeStatus] = {}
_TASK_OUTCOME_TO_TASK_STATUS: dict[Any, TaskStatus] = {}


def _ensure_mappings() -> None:
    if _TASK_OUTCOME_TO_RUNTIME:
        return
    from ..teammate import TeammateStatus

    _TASK_OUTCOME_TO_RUNTIME[TeammateStatus.REPORTING] = RuntimeStatus.COMPLETED
    _TASK_OUTCOME_TO_RUNTIME[TeammateStatus.TIMEOUT] = RuntimeStatus.TIMEOUT
    _TASK_OUTCOME_TO_RUNTIME[TeammateStatus.FAILED] = RuntimeStatus.FAILED
    _TASK_OUTCOME_TO_RUNTIME[TeammateStatus.CANCELLED] = RuntimeStatus.CANCELLED

    _TASK_OUTCOME_TO_TASK_STATUS[TeammateStatus.REPORTING] = TaskStatus.SUCCEEDED
    _TASK_OUTCOME_TO_TASK_STATUS[TeammateStatus.TIMEOUT] = TaskStatus.TIMED_OUT
    _TASK_OUTCOME_TO_TASK_STATUS[TeammateStatus.FAILED] = TaskStatus.FAILED
    _TASK_OUTCOME_TO_TASK_STATUS[TeammateStatus.CANCELLED] = TaskStatus.CANCELLED


class _TeammateExecutorAdapter:
    """将 TeammateRuntime 适配为 AgentExecutor 接口。

    execute(task) 通过 bus 向 teammate 发送任务，等待 future 结果，
    转换为 mcc_core.runtime.RuntimeResult 返回。
    """

    def __init__(self, teammate: Any, bus: Any, lead_name: str) -> None:
        self._teammate = teammate
        self._bus = bus
        self._lead_name = lead_name

    async def execute(
        self,
        task: RuntimeTask,
        **kwargs: Any,
    ) -> CoreRuntimeResult:
        _ensure_mappings()
        future: asyncio.Future[tuple[Any, str]] = asyncio.get_running_loop().create_future()
        self._teammate.set_task_future(task.task_id, future)
        await self._bus.send(
            self._lead_name,
            self._teammate.name,
            task.objective,
            kind="task_submitted",
            task_id=task.task_id,
        )
        self._teammate.notify()
        outcome, summary = await future
        task_status = _TASK_OUTCOME_TO_TASK_STATUS.get(outcome, TaskStatus.FAILED)
        now = datetime.now(UTC)
        return CoreRuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._teammate.runtime_id,
            agent_id=task.agent_id,
            status=task_status,
            summary=summary,
            started_at=now,
            finished_at=now,
        )

    async def close(self) -> None:
        pass


class PersistentRuntimeV2:
    """PersistentRuntimeV2：实现 mcc_core.runtime.protocols.AgentRuntime。

    通过 BoundedTaskQueue + WorkerPool + AgentExecutor 执行任务。
    通过 from_teammate() 提供 TeammateRuntime 向后兼容。
    """

    def __init__(
        self,
        executor: Any,
        *,
        max_concurrency: int = 1,
        queue_capacity: int = 0,
        runtime_id: str | None = None,
        agent_id: str = "",
    ) -> None:
        self._executor = executor
        self._runtime_id = runtime_id or f"persistent::{uuid4().hex[:12]}"
        self._agent_id = agent_id or self._runtime_id
        self._capabilities = PERSISTENT_CAPABILITIES
        self._max_concurrency = max_concurrency
        self._queue = BoundedTaskQueue(max_size=queue_capacity)
        self._worker_pool = WorkerPool(
            self._queue,
            executor,
            max_workers=max_concurrency,
            on_event=self._on_worker_event,
        )

        self._status = CoreRuntimeStatus.CREATED
        self._started_at: float = 0.0
        self._event_seq = 0
        self._event_listeners: list[asyncio.Queue] = []
        self._pending_sends: set[asyncio.Task] = set()
        self._task_futures: dict[str, asyncio.Future] = {}

        self._teammate: Any = None
        self._bus: Any = None
        self._lead_name: str = ""

    @classmethod
    def from_teammate(
        cls,
        teammate: Any,
        bus: Any,
        *,
        lead_name: str | None = None,
        runtime_id: str | None = None,
    ) -> PersistentRuntimeV2:
        from ..teammate import LEAD_NAME

        ln = lead_name or LEAD_NAME
        adapter = _TeammateExecutorAdapter(teammate, bus, ln)
        instance = cls(
            adapter,
            max_concurrency=1,
            runtime_id=runtime_id,
            agent_id=teammate.spec.name,
        )
        instance._teammate = teammate
        instance._bus = bus
        instance._lead_name = ln
        return instance

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
    def capabilities(self) -> Any:
        return self._capabilities

    @property
    def teammate(self) -> Any:
        return self._teammate

    @property
    def alive(self) -> bool:
        if self._teammate is not None:
            return self._teammate.alive
        return self._worker_pool.is_started

    async def start(self) -> None:
        self._started_at = time.monotonic()
        self._status = CoreRuntimeStatus.STARTING
        await self._emit(RuntimeEventType.RUNTIME_STARTED)

        if self._teammate is not None and not self._teammate.alive:
            await self._teammate.start()

        self._worker_pool.start()

        self._status = CoreRuntimeStatus.READY
        await self._emit(RuntimeEventType.RUNTIME_READY)

    async def submit(self, task: Any) -> Any:
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
        future = await self._queue.put(task)
        self._task_futures[task.task_id] = future
        return handle

    async def _on_worker_event(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        event_map = {
            "task_started": RuntimeEventType.TASK_STARTED,
            "task_completed": RuntimeEventType.TASK_COMPLETED,
            "task_failed": RuntimeEventType.TASK_FAILED,
        }
        rt_event_type = event_map.get(event_type)
        if rt_event_type is not None:
            await self._emit(
                rt_event_type,
                task_id=task_id,
                attempt_id=attempt_id,
                payload=payload or {},
            )

    async def cancel_task(
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None:
        future = self._task_futures.get(task_id)
        if future is not None and not future.done():
            future.cancel()

        if self._teammate is not None:
            self._send("cancel", reason or "cancel")
            self._teammate.notify()

        await self._emit(
            RuntimeEventType.TASK_CANCELLED,
            task_id=task_id,
            attempt_id=attempt_id,
            payload={"reason": reason} if reason else {},
        )

    async def pause(self) -> None:
        if self._teammate is not None:
            self._send("pause", "pause")
            self._teammate.notify()

    async def resume(self) -> None:
        if self._teammate is not None:
            self._send("resume", "resume")
            self._teammate.notify()

    async def stop(
        self,
        timeout: float | None = None,
        *,
        cancel_running: bool = False,
    ) -> None:
        self._status = CoreRuntimeStatus.STOPPING
        await self._emit(RuntimeEventType.RUNTIME_STOPPING)

        await self._worker_pool.stop(timeout=timeout or 5.0)

        if self._teammate is not None and self._teammate.alive:
            try:
                await self._teammate.shutdown(timeout=timeout or 5.0)
            except RuntimeStopTimeoutError:
                self._status = CoreRuntimeStatus.STOPPED
                await self._emit(RuntimeEventType.RUNTIME_STOPPED)
                for listener in self._event_listeners:
                    await listener.put(None)
                raise

        if self._teammate is None and hasattr(self._executor, "close"):
            await self._executor.close()

        self._status = CoreRuntimeStatus.STOPPED
        await self._emit(RuntimeEventType.RUNTIME_STOPPED)
        for listener in self._event_listeners:
            await listener.put(None)

    async def health_check(self) -> Any:
        uptime = time.monotonic() - self._started_at if self._started_at else 0.0
        state = HealthState.HEALTHY
        if self._status is CoreRuntimeStatus.FAILED:
            state = HealthState.UNHEALTHY
        elif self._status in (
            CoreRuntimeStatus.STOPPING,
            CoreRuntimeStatus.STOPPED,
        ):
            state = HealthState.UNKNOWN
        return RuntimeHealth(
            runtime_id=self._runtime_id,
            state=state,
            active_tasks=self._queue.pending,
            uptime_seconds=uptime,
        )

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> Any:
        return self._event_iterator(task_id=task_id, attempt_id=attempt_id)

    async def _event_iterator(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> Any:

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

    async def _emit(
        self,
        event_type: Any,
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

    def _send(self, kind: str, content: str) -> None:
        task = asyncio.create_task(
            self._bus.send(self._lead_name, self._teammate.name, content, kind=kind)
        )
        self._pending_sends.add(task)
        task.add_done_callback(self._pending_sends.discard)

    def inject_instruction(self, instruction: str) -> None:
        if self._teammate is None or not self._teammate.alive:
            return
        self._send("instruction", instruction)
        self._teammate.notify()

    @property
    def bus(self) -> Any:
        return self._bus

    def notify(self) -> None:
        if self._teammate is not None:
            self._teammate.notify()

    def cancel(self) -> None:
        if self._teammate is None or not self._teammate.alive:
            return
        self._send("cancel", "cancel")
        self._teammate.notify()

    def info(self) -> Any:
        if self._teammate is not None:
            return self._teammate.info()
        return None

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        """兼容 V1 接口：提交任务并返回 task_id。"""
        tid = task_id or f"task::{uuid4().hex[:12]}"
        rt_task = RuntimeTask(
            task_id=tid,
            attempt_id=f"{tid}:0",
            agent_id=self._agent_id,
            objective=task,
        )
        future = await self._queue.put(rt_task)
        self._task_futures[tid] = future
        return tid

    async def wait_task(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> RuntimeResult:
        """兼容 V1 接口：等待任务完成并返回 RuntimeResult。"""
        future = self._task_futures.get(task_id)
        if future is None:
            return _terminal_result(
                self._runtime_id,
                RuntimeStatus.FAILED,
                summary=f"unknown task_id: {task_id}",
                started_at=self._started_at,
            )
        try:
            if timeout is not None:
                result = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=timeout,
                )
            else:
                result = await asyncio.shield(future)
        except TimeoutError:
            raise WaitTimeoutError(
                self._runtime_id,
                timeout or 0.0,
            ) from None
        except asyncio.CancelledError:
            return _terminal_result(
                self._runtime_id,
                RuntimeStatus.CANCELLED,
                summary="cancelled",
                started_at=self._started_at,
            )

        if isinstance(result, CoreRuntimeResult):
            status = _TASK_STATUS_TO_RUNTIME.get(
                result.status,
                RuntimeStatus.FAILED,
            )
            return _terminal_result(
                self._runtime_id,
                status,
                summary=result.summary,
                started_at=self._started_at,
            )

        _ensure_mappings()
        outcome, summary = result
        status = _TASK_OUTCOME_TO_RUNTIME.get(outcome, RuntimeStatus.FAILED)
        return _terminal_result(
            self._runtime_id,
            status,
            summary=summary,
            started_at=self._started_at,
        )
