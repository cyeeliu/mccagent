"""新 Protocol Runtime 基类：共享状态机、事件分发、健康检查逻辑。

所有新 Protocol Runtime 实现继承 BaseRuntime，获得：
- 合法状态迁移验证（InvalidTransitionError）
- 事件队列与 subscribe_events 实现
- health_check 基础实现
- 生命周期时间追踪

Phase 4 交付物：新 Protocol Runtime 的公共基类。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

from mcc_core.runtime import (
    HealthState,
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeStatus,
    TaskHandle,
    is_valid_runtime_transition,
)
from mcc_core.runtime.errors import InvalidTransitionError, RuntimeNotReadyError
from mcc_core.runtime.identifiers import AgentId, RuntimeId
from mcc_core.runtime.task import RuntimeTask


class BaseRuntime:
    """新 Protocol Runtime 公共基类。

    子类需实现：
    - _on_start(): Runtime 启动后的初始化逻辑
    - _on_stop(): Runtime 停止前的清理逻辑
    - _on_submit(task): 任务提交后的处理逻辑
    - _on_cancel_task(task_id, attempt_id, reason): 任务取消逻辑
    - _on_pause() / _on_resume(): 暂停/恢复逻辑（可选）
    """

    def __init__(
        self,
        *,
        runtime_id: RuntimeId | None = None,
        agent_id: AgentId = "",
        capabilities: RuntimeCapabilities,
    ) -> None:
        self._runtime_id = runtime_id or f"rt::{uuid4().hex[:12]}"
        self._agent_id = agent_id or self._runtime_id
        self._capabilities = capabilities
        self._status = RuntimeStatus.CREATED
        self._started_at: float = 0.0
        self._event_queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self._event_seq = 0
        self._event_listeners: list[asyncio.Queue[RuntimeEvent]] = []
        self._active_tasks: int = 0
        self._lock = asyncio.Lock()

    @property
    def runtime_id(self) -> RuntimeId:
        return self._runtime_id

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return self._capabilities

    async def start(self) -> None:
        async with self._lock:
            self._transition(RuntimeStatus.STARTING)
            self._started_at = time.monotonic()
            await self._emit(RuntimeEventType.RUNTIME_STARTED)
            try:
                await self._on_start()
                self._transition(RuntimeStatus.READY)
                await self._emit(RuntimeEventType.RUNTIME_READY)
            except Exception:
                self._transition(RuntimeStatus.FAILED)
                await self._emit(RuntimeEventType.RUNTIME_FAILED)
                raise

    async def submit(self, task: RuntimeTask) -> TaskHandle:
        if self._status is not RuntimeStatus.READY:
            raise RuntimeNotReadyError(self._runtime_id, str(self._status))
        if not self._capabilities.supports_multiple_tasks and self._active_tasks > 0:
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
        await self._on_submit(task)
        return handle

    async def cancel_task(
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None:
        if not self._capabilities.supports_cancel_task:
            from mcc_core.runtime.errors import UnsupportedOperationError

            raise UnsupportedOperationError("cancel_task", self._runtime_id)
        await self._on_cancel_task(task_id, attempt_id, reason)
        await self._emit(
            RuntimeEventType.TASK_CANCELLED,
            task_id=task_id,
            attempt_id=attempt_id,
            payload={"reason": reason} if reason else {},
        )

    async def pause(self) -> None:
        if not self._capabilities.supports_pause:
            from mcc_core.runtime.errors import UnsupportedOperationError

            raise UnsupportedOperationError("pause", self._runtime_id)
        async with self._lock:
            self._transition(RuntimeStatus.PAUSING)
            await self._on_pause()
            self._transition(RuntimeStatus.PAUSED)

    async def resume(self) -> None:
        if not self._capabilities.supports_resume:
            from mcc_core.runtime.errors import UnsupportedOperationError

            raise UnsupportedOperationError("resume", self._runtime_id)
        async with self._lock:
            self._transition(RuntimeStatus.READY)
            await self._on_resume()

    async def stop(
        self,
        timeout: float | None = None,
        *,
        cancel_running: bool = False,
    ) -> None:
        async with self._lock:
            if self._status in (RuntimeStatus.STOPPED, RuntimeStatus.FAILED):
                return
            self._transition(RuntimeStatus.STOPPING)
            await self._emit(RuntimeEventType.RUNTIME_STOPPING)
            await self._on_stop(timeout=timeout, cancel_running=cancel_running)
            self._transition(RuntimeStatus.STOPPED)
            await self._emit(RuntimeEventType.RUNTIME_STOPPED)

    async def health_check(self) -> RuntimeHealth:
        if not self._capabilities.supports_health_check:
            from mcc_core.runtime.errors import UnsupportedOperationError

            raise UnsupportedOperationError("health_check", self._runtime_id)
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
            self._event_listeners.remove(queue)

    async def _on_start(self) -> None:
        pass

    async def _on_stop(
        self,
        *,
        timeout: float | None = None,
        cancel_running: bool = False,
    ) -> None:
        pass

    async def _on_submit(self, task: RuntimeTask) -> None:
        pass

    async def _on_cancel_task(
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None:
        pass

    async def _on_pause(self) -> None:
        pass

    async def _on_resume(self) -> None:
        pass

    def _transition(self, target: RuntimeStatus) -> None:
        if not is_valid_runtime_transition(self._status, target):
            raise InvalidTransitionError(str(self._status), str(target))
        self._status = target

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

    def _task_completed(self) -> None:
        if self._active_tasks > 0:
            self._active_tasks -= 1


__all__ = ["BaseRuntime"]
