"""P1-06 回归测试：AgentFactory 从 AgentDefinition 装配 Runtime/Executor。

验证 DefaultAgentFactory.create_runtime / create_executor 从
AgentDefinition 的 profile 装配 AgentLoop → AgentLoopExecutor → Runtime，
而非依赖 SessionContext 中预构建的对象。使用真实集成装配（MockLlm +
default_registry + ToolContext），不注入 mock。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from mcc_core import ToolContext, Usage
from mcc_core.runtime import (
    AgentDefinition,
    ExecutionProfile,
    ModelProfile,
    PermissionProfile,
    ToolProfile,
)
from mcc_core.runtime.enums import RuntimeKind
from mcc_harness.execution import DefaultAgentFactory, FactoryContext
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry

# -- 共享 fixture：真实集成装配环境 --


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def llm() -> MockLlm:
    return MockLlm([single_text_turn("ok", Usage(input_tokens=1, output_tokens=1))])


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def tool_context(work_dir: Path) -> ToolContext:
    return ToolContext(cwd=work_dir, session_id="p1-06", depth=0)


@pytest.fixture
def factory_context(llm, registry, tool_context, work_dir):
    return FactoryContext(
        llm=llm,
        registry=registry,
        tool_context=tool_context,
        work_dir=work_dir,
        config=None,
    )


def _make_definition(
    *,
    model_id: str = "test-model-001",
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_iterations: int = 25,
    runtime_kind: RuntimeKind = RuntimeKind.INTERACTIVE,
) -> AgentDefinition:
    """构造测试用 AgentDefinition，profile 参数可定制。"""
    return AgentDefinition(
        agent_id="agent-p1-06",
        name="p1-06-agent",
        role="worker",
        description="P1-06 regression agent",
        model_profile=ModelProfile(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        tool_profile=ToolProfile(),
        permission_profile=PermissionProfile(),
        execution_profile=ExecutionProfile(max_iterations=max_iterations),
        runtime_kind=runtime_kind,
    )


class TestFactoryContext:
    """FactoryContext 只含共享基础设施，不含预构建对象。"""

    def test_has_infrastructure_fields(self, factory_context):
        assert factory_context.llm is not None
        assert factory_context.registry is not None
        assert factory_context.tool_context is not None
        assert factory_context.work_dir is not None

    def test_has_no_prebuilt_agent_fields(self):
        fields = {f.name for f in dataclasses.fields(FactoryContext)}
        assert "agent_loop" not in fields
        assert "teammate" not in fields

    def test_optional_fields_default_none(self, tool_context, work_dir):
        ctx = FactoryContext(
            llm=object(),
            registry=object(),
            tool_context=tool_context,
            work_dir=work_dir,
            config=None,
        )
        assert ctx.hooks is None
        assert ctx.recorder is None
        assert ctx.coordinator is None
        assert ctx.bus is None


class TestCreateRuntimeFromDefinition:
    """create_runtime 从 AgentDefinition profile 装配 Runtime。"""

    @pytest.mark.asyncio
    async def test_interactive_returns_interactive_runtime(self, factory_context):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        factory = DefaultAgentFactory()
        definition = _make_definition(runtime_kind=RuntimeKind.INTERACTIVE)
        runtime = await factory.create_runtime(definition, factory_context)
        assert isinstance(runtime, InteractiveRuntimeV2)
        assert runtime.agent_id == "agent-p1-06"

    @pytest.mark.asyncio
    async def test_interactive_agent_uses_model_id(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition(model_id="custom-model-xyz")
        runtime = await factory.create_runtime(definition, factory_context)
        assert runtime.agent.model == "custom-model-xyz"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_id", "max_tokens", "temperature"),
        [
            ("model-a", 1024, 0.0),
            ("model-b", 4096, 0.5),
            ("model-c", 8192, 1.0),
        ],
    )
    async def test_interactive_propagates_model_profile(
        self,
        factory_context,
        model_id,
        max_tokens,
        temperature,
    ):
        factory = DefaultAgentFactory()
        definition = _make_definition(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        runtime = await factory.create_runtime(definition, factory_context)
        agent = runtime.agent
        assert agent.model == model_id
        assert agent.max_tokens == max_tokens
        assert agent.temperature == temperature

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_iterations", [5, 15, 50])
    async def test_interactive_propagates_execution_profile(
        self,
        factory_context,
        max_iterations,
    ):
        factory = DefaultAgentFactory()
        definition = _make_definition(max_iterations=max_iterations)
        runtime = await factory.create_runtime(definition, factory_context)
        assert runtime.agent.max_iterations == max_iterations

    @pytest.mark.asyncio
    async def test_task_returns_task_runtime(self, factory_context):
        from mcc_harness.runtime.task_runtime import TaskRuntime

        factory = DefaultAgentFactory()
        definition = _make_definition(runtime_kind=RuntimeKind.TASK)
        runtime = await factory.create_runtime(definition, factory_context)
        assert isinstance(runtime, TaskRuntime)
        assert runtime.agent_id == "agent-p1-06"

    @pytest.mark.asyncio
    async def test_task_executor_uses_model_id(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition(
            model_id="task-model-abc",
            runtime_kind=RuntimeKind.TASK,
        )
        runtime = await factory.create_runtime(definition, factory_context)
        assert runtime._executor._loop.model == "task-model-abc"

    @pytest.mark.asyncio
    async def test_task_propagates_max_iterations(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition(
            max_iterations=33,
            runtime_kind=RuntimeKind.TASK,
        )
        runtime = await factory.create_runtime(definition, factory_context)
        assert runtime._executor._loop.max_iterations == 33

    @pytest.mark.asyncio
    async def test_remote_raises_not_implemented(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition(runtime_kind=RuntimeKind.REMOTE)
        with pytest.raises(NotImplementedError, match="not supported"):
            await factory.create_runtime(definition, factory_context)

    @pytest.mark.asyncio
    async def test_without_llm_raises(self, registry, tool_context, work_dir):
        factory = DefaultAgentFactory()
        definition = _make_definition()
        context = FactoryContext(
            llm=None,
            registry=registry,
            tool_context=tool_context,
            work_dir=work_dir,
            config=None,
        )
        with pytest.raises(ValueError, match="llm is required"):
            await factory.create_runtime(definition, context)


class TestCreateExecutorFromDefinition:
    """create_executor 从 AgentDefinition profile 装配 AgentLoopExecutor。"""

    def test_returns_executor_with_model_from_definition(self, factory_context):
        from mcc_harness.execution import AgentLoopExecutor

        factory = DefaultAgentFactory()
        definition = _make_definition(model_id="exec-model-001")
        executor = factory.create_executor(definition, factory_context)
        assert isinstance(executor, AgentLoopExecutor)
        assert executor._loop.model == "exec-model-001"

    @pytest.mark.parametrize(
        ("model_id", "max_tokens", "temperature"),
        [
            ("exec-a", 512, 0.1),
            ("exec-b", 2048, 0.9),
        ],
    )
    def test_executor_propagates_model_profile(
        self,
        factory_context,
        model_id,
        max_tokens,
        temperature,
    ):
        factory = DefaultAgentFactory()
        definition = _make_definition(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        executor = factory.create_executor(definition, factory_context)
        loop = executor._loop
        assert loop.model == model_id
        assert loop.max_tokens == max_tokens
        assert loop.temperature == temperature

    def test_executor_propagates_execution_profile(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition(max_iterations=42)
        executor = factory.create_executor(definition, factory_context)
        assert executor._loop.max_iterations == 42

    def test_executor_carries_agent_id(self, factory_context):
        factory = DefaultAgentFactory()
        definition = _make_definition()
        executor = factory.create_executor(definition, factory_context)
        assert executor.agent_id == "agent-p1-06"


class TestBackwardCompatibility:
    """旧 create() 和 create_executor(agent_loop) 向后兼容。"""

    @pytest.mark.asyncio
    async def test_old_create_interactive_still_works(self, tmp_path):
        from mcc_core import ToolContext as TC
        from mcc_harness.execution import SessionContext
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        class StubLoop:
            model = "stub"
            registry = None

        ctx = TC(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            agent_loop=StubLoop(),
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a1",
            name="t",
            role="t",
            description="t",
            model_profile=ModelProfile(model_id="m1"),
        )
        runtime = await factory.create(definition)
        assert isinstance(runtime, InteractiveRuntimeV2)

    def test_old_create_executor_from_agent_loop_still_works(self):
        from mcc_harness.execution import AgentLoopExecutor

        class StubLoop:
            model = "stub"
            registry = None

        factory = DefaultAgentFactory()
        executor = factory.create_executor(
            StubLoop(),
            runtime_id="r1",
            agent_id="a1",
        )
        assert isinstance(executor, AgentLoopExecutor)
        assert executor.runtime_id == "r1"
        assert executor.agent_id == "a1"
