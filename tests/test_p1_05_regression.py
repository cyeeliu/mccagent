"""P1-05 回归测试：SubagentExecutor 从 AgentDefinition 装配而非委托 SubagentRunner。

验证：
- SubagentSpec.to_agent_definition() 正确映射字段到 AgentDefinition 的各 Profile；
- SubagentExecutor 提供 agent_factory + factory_context 时走新装配路径
  （AgentFactory.create_executor → AgentLoopExecutor），不再委托 SubagentRunner；
- 旧委托路径（仅传 runner/spec/ctx）向后兼容；
- 新路径使用隔离上下文（depth + 1）与 spec.model。

使用真实集成装配（MockLlm + default_registry + ToolContext），不注入 mock；
仅用 Stub 桩避免 LLM 依赖、用真实工厂子类记录装配参数。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core import ToolContext, Usage
from mcc_core.runtime import (
    AgentDefinition,
    ExecutionProfile,
    ModelProfile,
    TaskStatus,
    ToolProfile,
)
from mcc_core.runtime.enums import RuntimeKind
from mcc_core.runtime.task import RuntimeTask
from mcc_harness.execution import DefaultAgentFactory, FactoryContext
from mcc_harness.execution.subagent_executor import SubagentExecutor
from mcc_harness.subagent import SubagentResult, SubagentSpec, SubagentStatus
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry

# -- 共享 fixture：真实集成装配环境 --


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def llm() -> MockLlm:
    return MockLlm([single_text_turn("done", Usage(input_tokens=3, output_tokens=2))])


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def tool_context(work_dir: Path) -> ToolContext:
    return ToolContext(cwd=work_dir, session_id="p1-05", depth=0)


@pytest.fixture
def factory_context(llm, registry, tool_context, work_dir):
    return FactoryContext(
        llm=llm,
        registry=registry,
        tool_context=tool_context,
        work_dir=work_dir,
        config=None,
    )


@pytest.fixture
def agent_factory():
    return DefaultAgentFactory()


def _make_spec(**overrides) -> SubagentSpec:
    """构造测试用 SubagentSpec，字段可定制。"""
    defaults = dict(
        name="explorer",
        system="you are an explorer",
        model="claude-haiku-4-5",
        tools=["read_file", "list_dir"],
        max_iterations=8,
        max_tokens=1024,
        budget_tokens=50_000,
        timeout_secs=60.0,
    )
    defaults.update(overrides)
    return SubagentSpec(**defaults)


class _UnusedRunner:
    """新装配路径不应委托 runner，占位并守卫。"""

    async def run(self, *args: object, **kwargs: object) -> SubagentResult:
        raise AssertionError("factory path should not delegate to SubagentRunner")


class _RecordingFactory(DefaultAgentFactory):
    """真实工厂子类，记录 create_executor 装配参数供断言。"""

    def __init__(self) -> None:
        super().__init__()
        self.last_definition: object | None = None
        self.last_context: object | None = None
        self.last_executor: object | None = None

    def create_executor(
        self,
        definition_or_loop: object,
        context: object | None = None,
        **kwargs: object,
    ) -> object:
        self.last_definition = definition_or_loop
        self.last_context = context
        executor = super().create_executor(definition_or_loop, context, **kwargs)
        self.last_executor = executor
        return executor


def _runtime_task(agent_id: str = "explorer", objective: str = "explore") -> RuntimeTask:
    return RuntimeTask(
        task_id="t1",
        attempt_id="a1",
        agent_id=agent_id,
        objective=objective,
    )


# -- SubagentSpec.to_agent_definition() 转换测试 --


class TestSubagentSpecToAgentDefinition:
    """SubagentSpec.to_agent_definition() 字段映射到 AgentDefinition profiles。"""

    @pytest.mark.parametrize(
        ("model", "max_tokens"),
        [
            ("claude-haiku-4-5", 2048),
            ("claude-sonnet-4-6", 4096),
        ],
    )
    def test_model_profile_mapped(self, model: str, max_tokens: int) -> None:
        spec = _make_spec(model=model, max_tokens=max_tokens)
        definition = spec.to_agent_definition()
        assert isinstance(definition, AgentDefinition)
        assert isinstance(definition.model_profile, ModelProfile)
        assert definition.model_profile.model_id == model
        assert definition.model_profile.max_tokens == max_tokens

    def test_tool_profile_mapped_from_tools(self) -> None:
        spec = _make_spec(tools=["read_file", "list_dir", "grep"])
        definition = spec.to_agent_definition()
        assert isinstance(definition.tool_profile, ToolProfile)
        assert definition.tool_profile.allowed_tools == frozenset(
            {"read_file", "list_dir", "grep"},
        )

    def test_tool_profile_empty_when_no_tools(self) -> None:
        spec = _make_spec(tools=[])
        definition = spec.to_agent_definition()
        assert definition.tool_profile.allowed_tools == frozenset()

    @pytest.mark.parametrize(
        ("max_iterations", "budget_tokens", "timeout_secs"),
        [
            (5, 30_000, 30.0),
            (20, 150_000, 300.0),
        ],
    )
    def test_execution_profile_mapped(
        self,
        max_iterations: int,
        budget_tokens: int,
        timeout_secs: float,
    ) -> None:
        spec = _make_spec(
            max_iterations=max_iterations,
            budget_tokens=budget_tokens,
            timeout_secs=timeout_secs,
        )
        definition = spec.to_agent_definition()
        assert isinstance(definition.execution_profile, ExecutionProfile)
        assert definition.execution_profile.max_iterations == max_iterations
        assert definition.execution_profile.budget_tokens == budget_tokens
        assert definition.execution_profile.timeout_seconds == timeout_secs

    @pytest.mark.parametrize("name", ["code_explorer", "test_runner", "summarizer"])
    def test_identity_fields_mapped(self, name: str) -> None:
        spec = _make_spec(name=name, system="some system prompt")
        definition = spec.to_agent_definition()
        assert definition.agent_id == name
        assert definition.name == name

    def test_runtime_kind_is_task(self) -> None:
        spec = _make_spec()
        definition = spec.to_agent_definition()
        assert definition.runtime_kind is RuntimeKind.TASK

    def test_definition_is_frozen(self) -> None:
        spec = _make_spec()
        definition = spec.to_agent_definition()
        with pytest.raises((AttributeError, TypeError)):
            definition.agent_id = "mutated"  # type: ignore[misc]

    def test_default_presets_convertible(self) -> None:
        """预设库中每个 SubagentSpec 都能转换为 AgentDefinition。"""
        from mcc_harness.subagent import default_presets

        for preset_name, spec in default_presets().items():
            definition = spec.to_agent_definition()
            assert definition.model_profile.model_id == spec.model
            assert definition.agent_id == preset_name


# -- SubagentExecutor 新装配路径测试 --


class TestSubagentExecutorFactoryPath:
    """SubagentExecutor 通过 AgentFactory.create_executor 装配执行。"""

    async def test_factory_path_returns_succeeded_result(
        self,
        agent_factory: DefaultAgentFactory,
        factory_context: FactoryContext,
        tool_context: ToolContext,
    ) -> None:
        spec = _make_spec()
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            tool_context,
            agent_factory=agent_factory,
            factory_context=factory_context,
        )
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.summary == "done"

    async def test_factory_path_does_not_delegate_to_runner(
        self,
        agent_factory: DefaultAgentFactory,
        factory_context: FactoryContext,
        tool_context: ToolContext,
    ) -> None:
        """新路径不应调用 runner.run（_UnusedRunner.run 会 raise AssertionError）。"""
        spec = _make_spec()
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            tool_context,
            agent_factory=agent_factory,
            factory_context=factory_context,
        )
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.SUCCEEDED

    async def test_factory_path_uses_model_from_spec(
        self,
        factory_context: FactoryContext,
        tool_context: ToolContext,
    ) -> None:
        factory = _RecordingFactory()
        spec = _make_spec(model="custom-model-xyz")
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            tool_context,
            agent_factory=factory,
            factory_context=factory_context,
        )
        await executor.execute(_runtime_task())
        assert factory.last_executor is not None
        assembled_loop = factory.last_executor._loop  # type: ignore[attr-defined]
        assert assembled_loop.model == "custom-model-xyz"
        assert assembled_loop.max_tokens == 1024

    async def test_factory_path_uses_max_iterations_from_spec(
        self,
        factory_context: FactoryContext,
        tool_context: ToolContext,
    ) -> None:
        factory = _RecordingFactory()
        spec = _make_spec(max_iterations=42)
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            tool_context,
            agent_factory=factory,
            factory_context=factory_context,
        )
        await executor.execute(_runtime_task())
        assembled_loop = factory.last_executor._loop  # type: ignore[attr-defined]
        assert assembled_loop.max_iterations == 42

    async def test_factory_path_isolates_context_depth(
        self,
        llm: MockLlm,
        registry: object,
        work_dir: Path,
    ) -> None:
        """新路径为子代理创建隔离上下文（depth + 1）。"""
        parent_ctx = ToolContext(cwd=work_dir, session_id="parent", depth=2)
        factory = _RecordingFactory()
        ctx = FactoryContext(
            llm=llm,
            registry=registry,
            tool_context=parent_ctx,
            work_dir=work_dir,
            config=None,
        )
        spec = _make_spec(name="child")
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            parent_ctx,
            agent_factory=factory,
            factory_context=ctx,
        )
        await executor.execute(_runtime_task(agent_id="child"))
        assert factory.last_context is not None
        used_ctx = factory.last_context.tool_context  # type: ignore[attr-defined]
        assert used_ctx.depth == 3
        assert used_ctx.session_id == "parent::child"

    async def test_factory_path_closed_executor_returns_error(
        self,
        agent_factory: DefaultAgentFactory,
        factory_context: FactoryContext,
        tool_context: ToolContext,
    ) -> None:
        spec = _make_spec()
        executor = SubagentExecutor(
            _UnusedRunner(),
            spec,
            tool_context,
            agent_factory=agent_factory,
            factory_context=factory_context,
        )
        await executor.close()
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.FAILED
        assert "已关闭" in result.summary

    async def test_factory_path_only_factory_without_context_falls_back(
        self,
        tool_context: ToolContext,
    ) -> None:
        """仅提供 agent_factory 而无 factory_context 时走旧委托路径。"""

        class _StubRunner:
            async def run(
                self, spec: SubagentSpec, task: str, ctx: ToolContext, **kw: object
            ) -> SubagentResult:
                return SubagentResult(
                    name=spec.name,
                    summary="legacy",
                    status=SubagentStatus.COMPLETED,
                )

        spec = _make_spec()
        executor = SubagentExecutor(
            _StubRunner(),
            spec,
            tool_context,
            agent_factory=DefaultAgentFactory(),
            factory_context=None,
        )
        result = await executor.execute(_runtime_task())
        assert result.summary == "legacy"


# -- SubagentExecutor 旧委托路径向后兼容测试 --


class TestSubagentExecutorBackwardCompat:
    """不提供 agent_factory 时走旧 SubagentRunner 委托路径。"""

    async def test_legacy_path_maps_completed(self, tool_context: ToolContext) -> None:
        class _StubRunner:
            async def run(
                self, spec: SubagentSpec, task: str, ctx: ToolContext, **kw: object
            ) -> SubagentResult:
                return SubagentResult(
                    name=spec.name,
                    summary="legacy done",
                    status=SubagentStatus.COMPLETED,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    cost_usd=0.01,
                )

        spec = _make_spec()
        executor = SubagentExecutor(_StubRunner(), spec, tool_context)
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.summary == "legacy done"
        assert result.usage is not None
        assert result.usage.input_tokens == 10

    async def test_legacy_path_maps_timeout(self, tool_context: ToolContext) -> None:
        class _StubRunner:
            async def run(
                self, spec: SubagentSpec, task: str, ctx: ToolContext, **kw: object
            ) -> SubagentResult:
                return SubagentResult(
                    name=spec.name,
                    summary="timed out",
                    status=SubagentStatus.TIMEOUT,
                )

        spec = _make_spec()
        executor = SubagentExecutor(_StubRunner(), spec, tool_context)
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.TIMED_OUT

    async def test_legacy_path_error(self, tool_context: ToolContext) -> None:
        class _StubRunner:
            async def run(
                self, spec: SubagentSpec, task: str, ctx: ToolContext, **kw: object
            ) -> SubagentResult:
                return SubagentResult(
                    name=spec.name,
                    summary="boom",
                    status=SubagentStatus.ERROR,
                    error="something broke",
                )

        spec = _make_spec()
        executor = SubagentExecutor(_StubRunner(), spec, tool_context)
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.FAILED
        assert result.error is not None
        assert "something broke" in result.error.message

    async def test_legacy_path_close(self, tool_context: ToolContext) -> None:
        from mcc_harness.subagent import SubagentRunner

        spec = _make_spec()
        executor = SubagentExecutor(
            SubagentRunner.__new__(SubagentRunner),
            spec,
            tool_context,
        )
        await executor.close()
        result = await executor.execute(_runtime_task())
        assert result.status is TaskStatus.FAILED
        assert "已关闭" in result.summary
