"""SubagentExecutor：将 SubagentRunner 适配为 AgentExecutor。

Phase 13 迁移：TaskRuntime 依赖 AgentExecutor 协议，
SubagentExecutor 桥接 SubagentRunner.run() 到该协议，
使 SpawnSubagentTool 可通过 TaskRuntime 执行子代理。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from mcc_core import ToolContext
from mcc_core.runtime import RuntimeResult, TaskStatus, TokenUsage
from mcc_core.runtime.task import RuntimeTask

from mcc_harness.execution.cancellation import CancellationToken
from mcc_harness.subagent import SubagentResult, SubagentRunner, SubagentSpec, SubagentStatus

_SUBAGENT_STATUS_MAP: dict[SubagentStatus, TaskStatus] = {
    SubagentStatus.COMPLETED: TaskStatus.SUCCEEDED,
    SubagentStatus.TIMEOUT: TaskStatus.TIMED_OUT,
    SubagentStatus.CANCELLED: TaskStatus.CANCELLED,
    SubagentStatus.BUDGET_EXCEEDED: TaskStatus.FAILED,
    SubagentStatus.ERROR: TaskStatus.FAILED,
    SubagentStatus.RUNNING: TaskStatus.RUNNING,
}


class SubagentExecutor:
    """将 SubagentRunner 适配为 AgentExecutor 协议。

    持有 SubagentRunner + SubagentSpec + ToolContext，
    execute() 调用 runner.run() 并将 SubagentResult 映射为 RuntimeResult。

    当提供 agent_factory + factory_context 时，走装配路径：
    spec.to_agent_definition() → agent_factory.create_executor() → 直接执行，
    不再委托 SubagentRunner。未提供时走旧委托路径（向后兼容）。
    """

    def __init__(
        self,
        runner: SubagentRunner,
        spec: SubagentSpec,
        ctx: ToolContext,
        *,
        agent_factory: Any | None = None,
        factory_context: Any | None = None,
    ) -> None:
        self._runner = runner
        self._spec = spec
        self._ctx = ctx
        self._agent_factory = agent_factory
        self._factory_context = factory_context
        self._closed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RuntimeResult:
        if self._closed:
            return self._error_result(task, "executor_closed", "Executor 已关闭")

        if self._agent_factory is not None and self._factory_context is not None:
            return await self._execute_via_factory(
                task,
                emit=emit,
                cancellation=cancellation,
            )

        started_at = datetime.now(UTC)
        try:
            result = await self._runner.run(self._spec, task.objective, self._ctx)
        except asyncio.CancelledError:
            return self._build_result(
                task,
                SubagentResult(
                    name=self._spec.name,
                    summary="任务被取消",
                    status=SubagentStatus.CANCELLED,
                ),
                started_at,
            )
        except Exception as exc:
            return self._error_result(task, "execution_error", str(exc), started_at)

        return self._build_result(task, result, started_at)

    async def _execute_via_factory(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RuntimeResult:
        """通过 AgentFactory 从 AgentDefinition 装配 Executor 并执行。

        为子代理创建隔离上下文（depth + 1），用 spec.to_agent_definition()
        转换后经 agent_factory.create_executor 装配 AgentLoopExecutor，
        直接执行并返回 RuntimeResult，不委托 SubagentRunner。
        """
        from dataclasses import replace

        isolated_ctx = ToolContext(
            cwd=self._ctx.cwd,
            session_id=f"{self._ctx.session_id}::{self._spec.name}",
            depth=self._ctx.depth + 1,
        )
        context = replace(self._factory_context, tool_context=isolated_ctx)
        definition = self._spec.to_agent_definition()
        started_at = datetime.now(UTC)
        try:
            executor = self._agent_factory.create_executor(definition, context)
            return await executor.execute(task, emit=emit, cancellation=cancellation)
        except Exception as exc:
            return self._error_result(task, "execution_error", str(exc), started_at)

    def _build_result(
        self,
        task: RuntimeTask,
        subagent_result: SubagentResult,
        started_at: datetime,
    ) -> RuntimeResult:
        status = _SUBAGENT_STATUS_MAP.get(subagent_result.status, TaskStatus.FAILED)
        usage = None
        if subagent_result.usage is not None:
            usage = TokenUsage(
                input_tokens=subagent_result.usage.input_tokens,
                output_tokens=subagent_result.usage.output_tokens,
                cache_creation_input_tokens=subagent_result.usage.cache_creation_input_tokens,
                cache_read_input_tokens=subagent_result.usage.cache_read_input_tokens,
            )
        cost = Decimal(str(subagent_result.cost_usd)) if subagent_result.cost_usd else None
        error = None
        if status is TaskStatus.FAILED and subagent_result.error:
            from mcc_core.runtime import RuntimeErrorInfo

            error = RuntimeErrorInfo(
                code=subagent_result.status.value,
                message=subagent_result.error,
            )
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="",
            agent_id=task.agent_id,
            status=status,
            output={"summary": subagent_result.summary, "name": subagent_result.name},
            summary=subagent_result.summary,
            error=error,
            usage=usage,
            cost_usd=cost,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _error_result(
        self,
        task: RuntimeTask,
        code: str,
        message: str,
        started_at: datetime | None = None,
    ) -> RuntimeResult:
        from mcc_core.runtime import RuntimeErrorInfo

        now = datetime.now(UTC)
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="",
            agent_id=task.agent_id,
            status=TaskStatus.FAILED,
            summary=message,
            error=RuntimeErrorInfo(code=code, message=message),
            started_at=started_at or now,
            finished_at=now,
        )

    async def close(self) -> None:
        self._closed = True


__all__ = ["SubagentExecutor"]
