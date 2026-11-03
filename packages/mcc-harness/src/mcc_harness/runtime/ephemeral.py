"""EphemeralRuntime：短生命周期运行时，适配 SubagentRunner。

单次执行模型：start → run → 终态。不支持 pause/resume/inject_instruction
（短生命周期无运行时注入需求）。cancel 通过 asyncio.Task.cancel 实现。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from mcc_core import ToolContext

from ..subagent import (
    SubagentResult,
    SubagentRunner,
    SubagentSpec,
    SubagentStatus,
)
from .base import (
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    _terminal_result,
)

_SUBAGENT_STATUS_MAP: dict[SubagentStatus, RuntimeStatus] = {
    SubagentStatus.RUNNING: RuntimeStatus.RUNNING,
    SubagentStatus.COMPLETED: RuntimeStatus.COMPLETED,
    SubagentStatus.TIMEOUT: RuntimeStatus.TIMEOUT,
    SubagentStatus.CANCELLED: RuntimeStatus.CANCELLED,
    SubagentStatus.BUDGET_EXCEEDED: RuntimeStatus.BUDGET_EXCEEDED,
    SubagentStatus.ERROR: RuntimeStatus.FAILED,
}


class EphemeralRuntime:
    """短生命周期运行时：包装 SubagentRunner 的单次执行。

    通过 asyncio.Task 驱动 SubagentRunner.run，提供统一的
    start/stop/cancel/wait_done 接口。pause/resume/inject_instruction
    不适用于短生命周期模式，调用时抛 RuntimeError。
    """

    def __init__(
        self,
        runner: SubagentRunner,
        spec: SubagentSpec,
        ctx: ToolContext,
        *,
        channel: Any = None,
    ) -> None:
        self._runner = runner
        self._spec = spec
        self._ctx = ctx
        self._channel = channel
        self._task: asyncio.Task[SubagentResult] | None = None
        self._status = RuntimeStatus.CREATED
        self._result: RuntimeResult | None = None
        self._started_at: float = 0.0

    @property
    def runtime_id(self) -> str:
        return f"{self._ctx.session_id}::{self._spec.name}"

    @property
    def status(self) -> RuntimeStatus:
        if self._task is not None and self._task.done():
            return self._status
        if self._task is not None:
            return RuntimeStatus.RUNNING
        return self._status

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def config(self) -> RuntimeConfig:
        return RuntimeConfig(
            name=self._spec.name,
            mode=RuntimeMode.EPHEMERAL,
            system=self._spec.system,
            model=self._spec.model,
            tools=self._spec.tools,
            max_iterations=self._spec.max_iterations,
            max_tokens=self._spec.max_tokens,
            budget_tokens=self._spec.budget_tokens,
            timeout_secs=self._spec.timeout_secs,
        )

    async def start(self, task: str, *, task_id: str | None = None) -> None:
        if self._task is not None and not self._task.done():
            raise RuntimeError(f"ephemeral runtime {self.runtime_id} already running")
        self._status = RuntimeStatus.STARTING
        self._started_at = time.monotonic()
        self._result = None
        self._task = asyncio.create_task(self._run(task))

    async def _run(self, task: str) -> SubagentResult:
        self._status = RuntimeStatus.RUNNING
        return await self._runner.run(
            self._spec, task, self._ctx, channel=self._channel
        )

    async def stop(self, *, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=timeout)
        self._finalize()

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def pause(self) -> None:
        raise RuntimeError("ephemeral runtime does not support pause")

    def resume(self) -> None:
        raise RuntimeError("ephemeral runtime does not support resume")

    def inject_instruction(self, instruction: str) -> None:
        raise RuntimeError(
            "ephemeral runtime does not support runtime instruction injection"
        )

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        if self._task is None:
            return _terminal_result(
                self.runtime_id,
                self._status,
                summary="not started",
            )
        try:
            if timeout is not None:
                await asyncio.wait_for(self._task, timeout=timeout)
            else:
                await self._task
        except asyncio.CancelledError:
            pass
        except TimeoutError:
            return _terminal_result(
                self.runtime_id,
                RuntimeStatus.TIMEOUT,
                summary=f"wait_done timed out after {timeout}s",
                started_at=self._started_at,
            )
        return self._finalize()

    def _finalize(self) -> RuntimeResult:
        if self._result is not None:
            return self._result

        if self._task is None or not self._task.done():
            return _terminal_result(
                self.runtime_id,
                self._status,
                summary="not finished",
                started_at=self._started_at,
            )

        if self._task.cancelled():
            status = RuntimeStatus.CANCELLED
            summary = "cancelled"
            self._status = status
            self._result = _terminal_result(
                self.runtime_id,
                status,
                summary=summary,
                started_at=self._started_at,
            )
            return self._result

        exc = self._task.exception()
        if exc is not None:
            status = RuntimeStatus.FAILED
            summary = f"runtime failed: {exc}"
            error = str(exc)
            self._status = status
            self._result = _terminal_result(
                self.runtime_id,
                status,
                summary=summary,
                error=error,
                started_at=self._started_at,
            )
            return self._result

        sub: SubagentResult = self._task.result()
        status = _SUBAGENT_STATUS_MAP.get(sub.status, RuntimeStatus.FAILED)
        self._status = status
        self._result = RuntimeResult(
            runtime_id=self.runtime_id,
            status=status,
            summary=sub.summary,
            usage=sub.usage,
            cost_usd=sub.cost_usd,
            error=sub.error,
            started_at=sub.started_at or self._started_at,
            finished_at=sub.finished_at or time.monotonic(),
        )
        return self._result
