"""PR-10: AgentFactory 唯一装配入口收敛（P1-07）。

旧实现存在多轨创建路径：
- create() 对 INTERACTIVE/TASK 从 session_context 读取预构建 agent_loop，
  忽略 AgentDefinition 的 profile。
- create_runtime() 从 AgentDefinition 装配但不支持 PERSISTENT。
- build_interactive_runtime_v2(builder) 和 create_interactive_runtime(agent)
  绕过 AgentDefinition 直接接受预构建对象。

修复后：
- create_runtime() 成为唯一装配入口，支持 INTERACTIVE/TASK/PERSISTENT。
- create() 在 agent_loop 缺失时委托 create_runtime() 从 AgentDefinition 装配。
- _assemble_agent_loop() 使用 execution_profile.budget_tokens 设置 max_cost_usd。
- build_interactive_runtime_v2() 和 create_interactive_runtime() 标记 deprecated。
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from mcc_core import ToolContext
from mcc_core.runtime import AgentDefinition, ExecutionProfile, ModelProfile
from mcc_core.runtime.enums import RuntimeKind
from mcc_harness.execution import DefaultAgentFactory, FactoryContext, SessionContext

# -- 桩对象 --


class _StubAgentLoop:
    """轻量 AgentLoop 桩。"""

    def __init__(self, model: str = "stub-model") -> None:
        self.model = model
        self.registry = None


class _StubBuilder:
    """轻量 AgentLoopBuilder 桩。"""

    def __init__(self, agent: _StubAgentLoop | None = None) -> None:
        self._agent = agent or _StubAgentLoop()

    def build(self) -> _StubAgentLoop:
        return self._agent


class _StubTeammateSpec:
    """轻量 Teammate spec 桩。"""

    def __init__(self, name: str = "teammate-1") -> None:
        self.name = name
        self.role = "coder"
        self.system = "test"
        self.model = "test"
        self.tools: list[str] = []
        self.max_iterations = 20
        self.max_tokens = 4096
        self.budget_tokens = 150_000
        self.require_plan = False
        self.timeout_secs = 300.0


class _StubTeammate:
    """轻量 TeammateRuntime 桩。"""

    def __init__(self, name: str = "teammate-1") -> None:
        self.spec = _StubTeammateSpec(name)
        self._plan_store = None
        self._task_store = None
        self._base_dir = None
        self._worktree_enabled = False
        self._lead_name = "lead"

    @property
    def runtime_id(self) -> str:
        return f"teammate::{self.spec.name}"

    def build_agent_executor(self, cwd: Any = None) -> Any:
        class _StubExecutor:
            async def execute(self, task: Any, **kw: Any) -> Any: ...
            async def close(self) -> None: ...

        return _StubExecutor()


def _make_factory_context(
    tmp_path,
    *,
    teammate=None,
    bus=None,
) -> FactoryContext:
    """构建含 LLM 的 FactoryContext，供 create_runtime 装配。"""
    ctx = ToolContext(cwd=tmp_path, session_id="s1")
    return FactoryContext(
        llm=object(),  # 桩 LLM，仅用于构造 AgentLoop
        registry=None,
        tool_context=ctx,
        work_dir=tmp_path,
        config=None,
        teammate=teammate,
        bus=bus,
    )


def _make_definition(
    *,
    runtime_kind: RuntimeKind = RuntimeKind.INTERACTIVE,
    budget_tokens: int = 80_000,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id="a1",
        name="test",
        role="test",
        description="test agent",
        model_profile=ModelProfile(model_id="m1"),
        execution_profile=ExecutionProfile(budget_tokens=budget_tokens),
        runtime_kind=runtime_kind,
    )


# -- 测试 --


class TestPr10CreateRuntimeSupportsPersistent:
    """create_runtime() 支持 PERSISTENT runtime。"""

    @pytest.mark.asyncio
    async def test_create_runtime_persistent(self, tmp_path) -> None:
        """create_runtime() 从 FactoryContext.teammate + bus 装配 PERSISTENT。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        teammate = _StubTeammate()
        bus = object()
        context = _make_factory_context(tmp_path, teammate=teammate, bus=bus)
        factory = DefaultAgentFactory()
        definition = _make_definition(runtime_kind=RuntimeKind.PERSISTENT)

        runtime = await factory.create_runtime(definition, context)
        assert isinstance(runtime, PersistentRuntimeV2)

    @pytest.mark.asyncio
    async def test_create_runtime_persistent_without_teammate_raises(
        self,
        tmp_path,
    ) -> None:
        """create_runtime() PERSISTENT 缺少 teammate 时抛 ValueError。"""
        context = _make_factory_context(tmp_path)
        factory = DefaultAgentFactory()
        definition = _make_definition(runtime_kind=RuntimeKind.PERSISTENT)

        with pytest.raises(ValueError, match="teammate and bus are required"):
            await factory.create_runtime(definition, context)


class TestPr10CreateDelegatesToCreateRuntime:
    """create() 在 agent_loop 缺失时委托 create_runtime() 从 AgentDefinition 装配。"""

    @pytest.mark.asyncio
    async def test_create_interactive_without_agent_loop_assembles_from_definition(
        self,
        tmp_path,
    ) -> None:
        """create() INTERACTIVE 无 agent_loop → 从 AgentDefinition 装配。"""
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=object(),  # 提供 LLM 供装配
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            # agent_loop 故意不提供
        )
        factory = DefaultAgentFactory(session)
        definition = _make_definition(runtime_kind=RuntimeKind.INTERACTIVE)

        runtime = await factory.create(definition)
        assert isinstance(runtime, InteractiveRuntimeV2)
        assert runtime.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_create_task_without_agent_loop_assembles_from_definition(
        self,
        tmp_path,
    ) -> None:
        """create() TASK 无 agent_loop → 从 AgentDefinition 装配。"""
        from mcc_harness.runtime.task_runtime import TaskRuntime

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=object(),
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        factory = DefaultAgentFactory(session)
        definition = _make_definition(runtime_kind=RuntimeKind.TASK)

        runtime = await factory.create(definition)
        assert isinstance(runtime, TaskRuntime)
        assert runtime.agent_id == "a1"


class TestPr10AssembleAgentLoopUsesBudget:
    """_assemble_agent_loop() 使用 execution_profile.budget_tokens。"""

    @pytest.mark.asyncio
    async def test_budget_tokens_propagated_to_max_cost_usd(
        self,
        tmp_path,
    ) -> None:
        """execution_profile.budget_tokens → AgentLoop.max_cost_usd。"""
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        context = _make_factory_context(tmp_path)
        factory = DefaultAgentFactory()
        definition = _make_definition(budget_tokens=42_000)

        runtime = await factory.create_runtime(definition, context)
        assert isinstance(runtime, InteractiveRuntimeV2)
        assert runtime.agent.max_cost_usd == 42_000


class TestPr10DeprecationWarnings:
    """build_interactive_runtime_v2 / create_interactive_runtime 标记 deprecated。"""

    def test_build_interactive_runtime_v2_emits_deprecation_warning(self) -> None:
        factory = DefaultAgentFactory()
        builder = _StubBuilder()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            factory.build_interactive_runtime_v2(builder, agent_id="a1")
            assert any(issubclass(x.category, DeprecationWarning) for x in w), (
                "build_interactive_runtime_v2 should emit DeprecationWarning"
            )

    def test_create_interactive_runtime_emits_deprecation_warning(self) -> None:
        factory = DefaultAgentFactory()
        agent = _StubAgentLoop()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            factory.create_interactive_runtime(agent, agent_id="a1")
            assert any(issubclass(x.category, DeprecationWarning) for x in w), (
                "create_interactive_runtime should emit DeprecationWarning"
            )
