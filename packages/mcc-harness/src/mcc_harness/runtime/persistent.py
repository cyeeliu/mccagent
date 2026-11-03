"""PersistentRuntime：常驻运行时，适配 TeammateRuntime。

循环执行模型：start → (idle | running | paused)* → stop → shutdown。
支持 cancel/pause/resume/inject_instruction 全部控制信号，
通过 MessageBus 向 teammate 发送控制消息。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..teammate import LEAD_NAME, TeammateRuntime, TeammateStatus
from .base import (
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    _terminal_result,
)

_TEAMMATE_STATUS_MAP: dict[TeammateStatus, RuntimeStatus] = {
    TeammateStatus.CREATED: RuntimeStatus.CREATED,
    TeammateStatus.STARTING: RuntimeStatus.STARTING,
    TeammateStatus.IDLE: RuntimeStatus.IDLE,
    TeammateStatus.ASSIGNED: RuntimeStatus.RUNNING,
    TeammateStatus.PLANNING: RuntimeStatus.RUNNING,
    TeammateStatus.WAITING_APPROVAL: RuntimeStatus.RUNNING,
    TeammateStatus.RUNNING: RuntimeStatus.RUNNING,
    TeammateStatus.REPORTING: RuntimeStatus.RUNNING,
    TeammateStatus.FAILED: RuntimeStatus.FAILED,
    TeammateStatus.TIMEOUT: RuntimeStatus.TIMEOUT,
    TeammateStatus.CANCELLED: RuntimeStatus.CANCELLED,
    TeammateStatus.RESTARTING: RuntimeStatus.STARTING,
    TeammateStatus.SHUTTING_DOWN: RuntimeStatus.STOPPING,
    TeammateStatus.SHUTDOWN: RuntimeStatus.SHUTDOWN,
}


class PersistentRuntime:
    """常驻运行时：包装 TeammateRuntime 的工作循环。

    通过 MessageBus 发送控制消息（cancel/pause/resume/instruction），
    通过 TeammateRuntime.start/shutdown 管理生命周期。
    wait_done 等待 teammate 到达 SHUTDOWN 终态。
    """

    def __init__(
        self,
        teammate: TeammateRuntime,
        bus: Any,
        *,
        lead_name: str = LEAD_NAME,
    ) -> None:
        self._teammate = teammate
        self._bus = bus
        self._lead_name = lead_name
        self._started_at: float = 0.0
        self._final_result: RuntimeResult | None = None

    @property
    def runtime_id(self) -> str:
        return self._teammate.name

    @property
    def status(self) -> RuntimeStatus:
        raw = self._teammate.status
        try:
            ts = TeammateStatus(raw)
        except ValueError:
            return RuntimeStatus.RUNNING
        return _TEAMMATE_STATUS_MAP.get(ts, RuntimeStatus.RUNNING)

    @property
    def alive(self) -> bool:
        return self._teammate.alive

    @property
    def config(self) -> RuntimeConfig:
        spec = self._teammate.spec
        return RuntimeConfig(
            name=spec.name,
            mode=RuntimeMode.PERSISTENT,
            system=spec.system,
            model=spec.model,
            tools=spec.tools,
            max_iterations=spec.max_iterations,
            max_tokens=spec.max_tokens,
            budget_tokens=spec.budget_tokens,
            timeout_secs=spec.timeout_secs,
            require_plan=spec.require_plan,
        )

    async def start(self, task: str, *, task_id: str | None = None) -> None:
        if self._teammate.alive:
            raise RuntimeError(
                f"persistent runtime {self.runtime_id} already running"
            )
        self._started_at = time.monotonic()
        self._final_result = None
        await self._teammate.start(task, task_id=task_id)

    async def stop(self, *, timeout: float = 5.0) -> None:
        await self._teammate.shutdown()

    def cancel(self) -> None:
        if not self._teammate.alive:
            return
        self._send("cancel", "cancel")
        self._teammate.notify()

    def pause(self) -> None:
        if not self._teammate.alive:
            return
        self._send("pause", "pause")
        self._teammate.notify()

    def resume(self) -> None:
        if not self._teammate.alive:
            return
        self._send("resume", "resume")
        self._teammate.notify()

    def inject_instruction(self, instruction: str) -> None:
        if not self._teammate.alive:
            return
        self._send("instruction", instruction)
        self._teammate.notify()

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        if self._final_result is not None:
            return self._final_result

        task = self._teammate._task
        if task is None:
            return _terminal_result(
                self.runtime_id,
                RuntimeStatus.SHUTDOWN,
                summary="not started",
                started_at=self._started_at,
            )

        try:
            if timeout is not None:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            else:
                await asyncio.shield(task)
        except TimeoutError:
            return _terminal_result(
                self.runtime_id,
                self.status,
                summary=f"wait_done timed out after {timeout}s",
                started_at=self._started_at,
            )
        except asyncio.CancelledError:
            pass

        return self._finalize()

    def _finalize(self) -> RuntimeResult:
        if self._final_result is not None:
            return self._final_result

        status = self.status
        if status == RuntimeStatus.SHUTDOWN:
            summary = "shutdown"
        elif status == RuntimeStatus.FAILED:
            summary = "teammate failed"
        elif status == RuntimeStatus.CANCELLED:
            summary = "cancelled"
        else:
            summary = f"stopped (status={status.value})"
            status = RuntimeStatus.SHUTDOWN

        self._final_result = _terminal_result(
            self.runtime_id,
            status,
            summary=summary,
            started_at=self._started_at,
        )
        return self._final_result

    def _send(self, kind: str, content: str) -> None:
        asyncio.create_task(
            self._bus.send(self._lead_name, self.runtime_id, content, kind=kind)
        )
