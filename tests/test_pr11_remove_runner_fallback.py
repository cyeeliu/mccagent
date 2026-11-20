"""PR-11: 删除 SubagentRunner fallback（P1-08）。

SubagentExecutor 不再保留 SubagentRunner 旧执行分支。
缺少 agent_factory / factory_context 时 fail fast，
不再回退到 runner.run()。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from mcc_core import ToolContext
from mcc_core.runtime import RuntimeResult, TaskStatus
from mcc_core.runtime.task import RuntimeTask
from mcc_harness.execution import DefaultAgentFactory, FactoryContext
from mcc_harness.execution.subagent_executor import SubagentExecutor
from mcc_harness.subagent import SubagentSpec

# -- 桩对象 --


class _StubAgentLoop:
    """轻量 AgentLoop 桩。"""

    def __init__(self, model: str = "stub-model") -> None:
        self.model = model
        self.registry = None
        self.max_cost_usd = 0.0


class _StubExecutor:
    """桩 AgentExecutor，记录 execute 调用。"""

    def __init__(self) -> None:
        self.executed = False

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.executed = True
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="",
            agent_id=task.agent_id,
            status=TaskStatus.SUCCEEDED,
            summary="stub result",
            started_at=now,
            finished_at=now,
        )

    async def close(self) -> None:
        pass


class _StubFactory:
    """桩 AgentFactory，返回预置 Executor。"""

    def __init__(self, executor: _StubExecutor | None = None) -> None:
        self._executor = executor or _StubExecutor()
        self.create_executor_calls: list[tuple[Any, Any]] = []

    def create_executor(
        self,
        definition_or_loop: Any,
        context: Any | None = None,
        *,
        runtime_id: str = "",
        agent_id: str = "",
    ) -> _StubExecutor:
        self.create_executor_calls.append((definition_or_loop, context))
        return self._executor


def _make_spec(name: str = "coder") -> SubagentSpec:
    return SubagentSpec(
        name=name,
        system="test system",
        model="test-model",
        tools=[],
        max_iterations=20,
        max_tokens=4096,
        budget_tokens=80_000,
        timeout_secs=300.0,
    )


def _make_task() -> RuntimeTask:
    return RuntimeTask(
        task_id="t1",
        attempt_id="a1",
        agent_id="coder",
        objective="do something",
    )


def _make_factory_context(tmp_path) -> FactoryContext:
    ctx = ToolContext(cwd=tmp_path, session_id="s1")
    return FactoryContext(
        llm=object(),
        registry=None,
        tool_context=ctx,
        work_dir=tmp_path,
        config=None,
    )


# -- 测试 --


class TestPr11NoRunnerFallback:
    """SubagentExecutor 不再回退到 runner.run()。"""

    def test_execute_source_has_no_runner_run(self) -> None:
        """SubagentExecutor.execute 源码不含 _runner.run 调用。"""
        source = inspect.getsource(SubagentExecutor.execute)
        assert "_runner.run" not in source, (
            "SubagentExecutor.execute should not call _runner.run"
        )

    @pytest.mark.asyncio
    async def test_missing_factory_raises_value_error(self, tmp_path) -> None:
        """缺少 agent_factory / factory_context 时 fail fast。"""
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        spec = _make_spec()
        executor = SubagentExecutor(
            spec=spec,
            ctx=ctx,
            agent_factory=None,
            factory_context=None,
        )
        task = _make_task()
        with pytest.raises(ValueError, match="agent_factory.*factory_context"):
            await executor.execute(task)

    @pytest.mark.asyncio
    async def test_runner_not_required(self, tmp_path) -> None:
        """runner 参数不再是必需的。"""
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        spec = _make_spec()
        factory = _StubFactory()
        factory_ctx = _make_factory_context(tmp_path)
        executor = SubagentExecutor(
            spec=spec,
            ctx=ctx,
            agent_factory=factory,
            factory_context=factory_ctx,
        )
        task = _make_task()
        result = await executor.execute(task)
        assert result.status is TaskStatus.SUCCEEDED


class TestPr11FactoryPathOnly:
    """SubagentExecutor 只走 factory 装配路径。"""

    @pytest.mark.asyncio
    async def test_factory_path_executed(self, tmp_path) -> None:
        """提供 agent_factory + factory_context 时走装配路径。"""
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        spec = _make_spec()
        stub_executor = _StubExecutor()
        factory = _StubFactory(stub_executor)
        factory_ctx = _make_factory_context(tmp_path)
        executor = SubagentExecutor(
            spec=spec,
            ctx=ctx,
            agent_factory=factory,
            factory_context=factory_ctx,
        )
        task = _make_task()
        await executor.execute(task)
        assert stub_executor.executed, "factory path should be executed"
        assert len(factory.create_executor_calls) == 1

    @pytest.mark.asyncio
    async def test_isolated_context_depth_incremented(self, tmp_path) -> None:
        """factory 路径创建隔离上下文 depth + 1。"""
        ctx = ToolContext(cwd=tmp_path, session_id="s1", depth=2)
        spec = _make_spec(name="explorer")
        factory = _StubFactory()
        factory_ctx = _make_factory_context(tmp_path)
        executor = SubagentExecutor(
            spec=spec,
            ctx=ctx,
            agent_factory=factory,
            factory_context=factory_ctx,
        )
        task = _make_task()
        await executor.execute(task)
        called_def, called_ctx = factory.create_executor_calls[0]
        assert called_ctx.tool_context.depth == 3
        assert called_ctx.tool_context.session_id == "s1::explorer"


class TestPr11SpawnSubagentToolPassesFactory:
    """SpawnSubagentTool 传递 agent_factory / factory_context 给 SubagentExecutor。"""

    def test_spawn_tool_accepts_factory_params(self) -> None:
        """SpawnSubagentTool.__init__ 接受 agent_factory / factory_context。"""
        from mcc_harness.subagent import SpawnSubagentTool

        sig = inspect.signature(SpawnSubagentTool.__init__)
        params = set(sig.parameters.keys())
        assert "agent_factory" in params, (
            "SpawnSubagentTool should accept agent_factory"
        )
        assert "factory_context" in params, (
            "SpawnSubagentTool should accept factory_context"
        )
