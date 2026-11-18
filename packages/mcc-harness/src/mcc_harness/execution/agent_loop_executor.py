"""AgentLoopExecutor：将 AgentLoop 封装为 AgentExecutor 适配层。

Phase 2 过渡实现：内部委托现有 AgentLoop，不改变其执行逻辑。
Runtime 不再直接依赖 AgentLoop 类型，而是通过 AgentExecutor 接口使用。

迁移完成后（Phase 4+），AgentLoop 的职责将逐步拆分到 Executor 内部组件。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from mcc_core.runtime import (
    RuntimeResult,
    TaskStatus,
    TokenUsage,
)
from mcc_core.runtime.task import RuntimeTask

from mcc_harness.agent import AgentLoop, AgentRun
from mcc_harness.execution.cancellation import CancellationToken
from mcc_harness.execution.context import ExecutionContext
from mcc_harness.execution.event_adapter import EventAdapter


class AgentLoopExecutor:
    def __init__(
        self,
        agent_loop: AgentLoop,
        *,
        runtime_id: str = "",
        agent_id: str = "",
    ) -> None:
        self._loop = agent_loop
        self._runtime_id = runtime_id or f"executor-{id(agent_loop)}"
        self._agent_id = agent_id
        self._closed = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RuntimeResult:
        if self._closed:
            return self._error_result(task, "executor_closed", "Executor 已关闭")

        ctx = ExecutionContext(task=task)
        cancel_event = cancellation.to_asyncio_event() if cancellation else None

        adapter = EventAdapter(
            runtime_id=self._runtime_id,
            agent_id=self._agent_id or task.agent_id,
            task_id=task.task_id,
            attempt_id=task.attempt_id,
        )

        try:
            run_result = await self._loop.run_streaming(
                task.objective,
                ctx.event_queue,
                instruction_queue=ctx.instruction_queue,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            return self._cancelled_result(task, ctx)
        except Exception as exc:
            return self._error_result(task, "execution_error", str(exc))

        if cancel_event is not None and cancel_event.is_set():
            return self._cancelled_result(task, ctx)

        if emit is not None:
            await self._drain_events(ctx, adapter, emit)

        return self._success_result(task, run_result, ctx)

    async def _drain_events(
        self,
        ctx: ExecutionContext,
        adapter: EventAdapter,
        emit: Any,
    ) -> None:
        while not ctx.event_queue.empty():
            try:
                agent_event = ctx.event_queue.get_nowait()
                runtime_event = adapter.adapt(agent_event)
                await emit.emit(runtime_event)
            except asyncio.QueueEmpty:
                break

    def _success_result(
        self,
        task: RuntimeTask,
        run: AgentRun,
        ctx: ExecutionContext,
    ) -> RuntimeResult:
        usage = TokenUsage(
            input_tokens=run.total_usage.input_tokens,
            output_tokens=run.total_usage.output_tokens,
            cache_creation_input_tokens=run.total_usage.cache_creation_input_tokens,
            cache_read_input_tokens=run.total_usage.cache_read_input_tokens,
        )
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
            agent_id=self._agent_id or task.agent_id,
            status=TaskStatus.SUCCEEDED,
            output={"final_text": run.final_text, "iterations": run.iterations},
            summary=run.final_text[:500] if run.final_text else "",
            usage=usage,
            cost_usd=Decimal(str(run.cost_usd)) if run.cost_usd else None,
            started_at=ctx.started_at,
            finished_at=datetime.now(UTC),
        )

    def _cancelled_result(self, task: RuntimeTask, ctx: ExecutionContext) -> RuntimeResult:
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
            agent_id=self._agent_id or task.agent_id,
            status=TaskStatus.CANCELLED,
            summary="任务被取消",
            started_at=ctx.started_at,
            finished_at=datetime.now(UTC),
        )

    def _error_result(
        self,
        task: RuntimeTask,
        code: str,
        message: str,
    ) -> RuntimeResult:
        from mcc_core.runtime import RuntimeErrorInfo
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
            agent_id=self._agent_id or task.agent_id,
            status=TaskStatus.FAILED,
            summary=message,
            error=RuntimeErrorInfo(code=code, message=message),
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

    async def close(self) -> None:
        self._closed = True


__all__ = ["AgentLoopExecutor"]
