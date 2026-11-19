"""统一 Runtime 与 Executor 协议（迁移方案 §4, §5）。

AgentRuntime Protocol 是所有 Runtime 实现的统一契约。
AgentExecutor Protocol 是执行单个 RuntimeTask 的统一接口。

Phase 1 只定义协议，不改变现有执行流。
Phase 2 实现 AgentExecutor，Phase 4 重写 Runtime 实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from mcc_core.runtime.capabilities import RuntimeCapabilities
from mcc_core.runtime.definition import AgentDefinition
from mcc_core.runtime.enums import RuntimeStatus
from mcc_core.runtime.event import RuntimeEvent
from mcc_core.runtime.handle import TaskHandle
from mcc_core.runtime.health import RuntimeHealth
from mcc_core.runtime.identifiers import AgentId, RuntimeId
from mcc_core.runtime.task import RuntimeTask


@runtime_checkable
class EventEmitter(Protocol):
    async def emit(self, event: RuntimeEvent) -> None: ...


@runtime_checkable
class CancellationToken(Protocol):
    @property
    def is_cancelled(self) -> bool: ...

    def cancel(self, reason: str | None = None) -> None: ...

    async def wait(self) -> None: ...


@runtime_checkable
class AgentRuntime(Protocol):
    @property
    def runtime_id(self) -> RuntimeId: ...

    @property
    def agent_id(self) -> AgentId: ...

    @property
    def status(self) -> RuntimeStatus: ...

    @property
    def capabilities(self) -> RuntimeCapabilities: ...

    async def start(self) -> None: ...

    async def submit(self, task: RuntimeTask) -> TaskHandle: ...

    async def cancel_task(
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def stop(
        self,
        timeout: float | None = None,
        *,
        cancel_running: bool = False,
    ) -> None: ...

    async def health_check(self) -> RuntimeHealth: ...

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]: ...


@runtime_checkable
class AgentExecutor(Protocol):
    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: EventEmitter | None = None,
        cancellation: CancellationToken | None = None,
    ) -> object: ...

    async def close(self) -> None: ...


@runtime_checkable
class AgentFactory(Protocol):
    async def create(
        self,
        definition: AgentDefinition,
        *,
        session_context: object | None = None,
    ) -> AgentRuntime: ...


__all__ = [
    "EventEmitter",
    "CancellationToken",
    "AgentRuntime",
    "AgentExecutor",
    "AgentFactory",
]
