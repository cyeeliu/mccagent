"""DefaultAgentFactory 单元测试。

验证 Factory 正确收拢 AgentLoop → AgentLoopExecutor → Runtime 的构造链。
"""

from __future__ import annotations

from typing import Any

import pytest
from mcc_harness.execution import DefaultAgentFactory, SessionContext


class _StubAgentLoop:
    """轻量 AgentLoop 桩，避免 LLM 依赖。"""

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
        """返回桩 executor，供 from_executor 路径使用。"""

        class _StubExecutor:
            async def execute(self, task: Any, **kw: Any) -> Any: ...
            async def close(self) -> None: ...

        return _StubExecutor()


class TestDefaultAgentFactory:
    """DefaultAgentFactory 核心行为。"""

    def test_create_without_session(self) -> None:
        factory = DefaultAgentFactory()
        assert factory.session is None

    def test_create_with_session(self, tmp_path) -> None:
        import pytest
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        factory = DefaultAgentFactory(session)
        assert factory.session is session

    def test_build_interactive_runtime_v2_returns_runtime_with_agent(self) -> None:
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        factory = DefaultAgentFactory()
        builder = _StubBuilder()
        runtime = factory.build_interactive_runtime_v2(
            builder,
            session_id="s1",
            agent_id="main",
        )
        assert isinstance(runtime, InteractiveRuntimeV2)
        assert runtime.agent_id == "main"
        assert runtime.agent is builder._agent

    def test_build_interactive_runtime_v2_generates_unique_ids(self) -> None:
        factory = DefaultAgentFactory()
        rt1 = factory.build_interactive_runtime_v2(_StubBuilder())
        rt2 = factory.build_interactive_runtime_v2(_StubBuilder())
        assert rt1.runtime_id != rt2.runtime_id

    def test_create_interactive_runtime_from_agent(self) -> None:
        factory = DefaultAgentFactory()
        agent = _StubAgentLoop()
        runtime = factory.create_interactive_runtime(
            agent,
            session_id="s2",
            agent_id="sub",
        )
        assert runtime.agent is agent
        assert runtime.session_id == "s2"

    def test_create_executor_wraps_agent_loop(self) -> None:
        factory = DefaultAgentFactory()
        agent = _StubAgentLoop()
        executor = factory.create_executor(
            agent,
            runtime_id="r1",
            agent_id="a1",
        )
        assert executor.runtime_id == "r1"
        assert executor.agent_id == "a1"
        assert not executor.is_closed

    @pytest.mark.asyncio
    async def test_create_interactive_from_definition(self, tmp_path) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_core.runtime.enums import RuntimeKind
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent_loop = _StubAgentLoop()
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            agent_loop=agent_loop,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a1",
            name="test",
            role="test",
            description="test agent",
            model_profile=ModelProfile(model_id="m1"),
            runtime_kind=RuntimeKind.INTERACTIVE,
        )
        runtime = await factory.create(definition)
        assert isinstance(runtime, InteractiveRuntimeV2)
        assert runtime.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_create_task_from_definition(self, tmp_path) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_core.runtime.enums import RuntimeKind
        from mcc_harness.runtime.task_runtime import TaskRuntime

        agent_loop = _StubAgentLoop()
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            agent_loop=agent_loop,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a2",
            name="task-agent",
            role="worker",
            description="task agent",
            model_profile=ModelProfile(model_id="m1"),
            runtime_kind=RuntimeKind.TASK,
        )
        runtime = await factory.create(definition)
        assert isinstance(runtime, TaskRuntime)
        assert runtime.agent_id == "a2"

    @pytest.mark.asyncio
    async def test_create_persistent_from_definition(self, tmp_path) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_core.runtime.enums import RuntimeKind
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        teammate = _StubTeammate()
        bus = object()
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            teammate=teammate,
            bus=bus,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a3",
            name="persistent-agent",
            role="teammate",
            description="persistent agent",
            model_profile=ModelProfile(model_id="m1"),
            runtime_kind=RuntimeKind.PERSISTENT,
        )
        runtime = await factory.create(definition)
        assert isinstance(runtime, PersistentRuntimeV2)

    @pytest.mark.asyncio
    async def test_create_remote_raises_not_implemented(self, tmp_path) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_core.runtime.enums import RuntimeKind

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a4",
            name="remote",
            role="remote",
            description="remote agent",
            model_profile=ModelProfile(model_id="m1"),
            runtime_kind=RuntimeKind.REMOTE,
        )
        with pytest.raises(NotImplementedError, match="not supported"):
            await factory.create(definition)

    @pytest.mark.asyncio
    async def test_create_without_session_raises_value_error(self) -> None:
        from mcc_core.runtime import AgentDefinition, ModelProfile

        factory = DefaultAgentFactory()
        definition = AgentDefinition(
            agent_id="a1",
            name="test",
            role="test",
            description="test agent",
            model_profile=ModelProfile(model_id="m1"),
        )
        with pytest.raises(ValueError, match="session_context is required"):
            await factory.create(definition)

    @pytest.mark.asyncio
    async def test_create_interactive_without_agent_loop_raises(
        self,
        tmp_path,
    ) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a1",
            name="test",
            role="test",
            description="test agent",
            model_profile=ModelProfile(model_id="m1"),
        )
        with pytest.raises(ValueError, match="agent_loop is required"):
            await factory.create(definition)

    @pytest.mark.asyncio
    async def test_create_persistent_without_teammate_raises(
        self,
        tmp_path,
    ) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_core.runtime.enums import RuntimeKind

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        factory = DefaultAgentFactory(session)
        definition = AgentDefinition(
            agent_id="a1",
            name="test",
            role="test",
            description="test agent",
            model_profile=ModelProfile(model_id="m1"),
            runtime_kind=RuntimeKind.PERSISTENT,
        )
        with pytest.raises(ValueError, match="teammate and bus are required"):
            await factory.create(definition)

    @pytest.mark.asyncio
    async def test_create_with_explicit_session_overrides_default(
        self,
        tmp_path,
    ) -> None:
        from mcc_core import ToolContext
        from mcc_core.runtime import AgentDefinition, ModelProfile
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent_loop = _StubAgentLoop()
        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        explicit_session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            agent_loop=agent_loop,
        )
        factory = DefaultAgentFactory()
        definition = AgentDefinition(
            agent_id="a1",
            name="test",
            role="test",
            description="test agent",
            model_profile=ModelProfile(model_id="m1"),
        )
        runtime = await factory.create(
            definition,
            session_context=explicit_session,
        )
        assert isinstance(runtime, InteractiveRuntimeV2)


class TestSessionContext:
    """SessionContext 数据类。"""

    def test_required_fields(self, tmp_path) -> None:
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=object(),
            registry=object(),
            tool_context=ctx,
            work_dir=tmp_path,
            config=object(),
        )
        assert session.llm is not None
        assert session.registry is not None
        assert session.tool_context is ctx
        assert session.hooks is None
        assert session.recorder is None
        assert session.coordinator is None

    def test_optional_fields(self, tmp_path) -> None:
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        hooks = object()
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            hooks=hooks,
            recorder=object(),
            coordinator=object(),
        )
        assert session.hooks is hooks
        assert session.recorder is not None
        assert session.coordinator is not None

    def test_v2_fields_default_none(self, tmp_path) -> None:
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
        )
        assert session.agent_loop is None
        assert session.teammate is None
        assert session.bus is None

    def test_v2_fields_set(self, tmp_path) -> None:
        from mcc_core import ToolContext

        ctx = ToolContext(cwd=tmp_path, session_id="s1")
        agent_loop = _StubAgentLoop()
        teammate = _StubTeammate()
        bus = object()
        session = SessionContext(
            llm=None,
            registry=None,
            tool_context=ctx,
            work_dir=tmp_path,
            config=None,
            agent_loop=agent_loop,
            teammate=teammate,
            bus=bus,
        )
        assert session.agent_loop is agent_loop
        assert session.teammate is teammate
        assert session.bus is bus
