"""AgentFactory：收拢 AgentLoop / Executor / Runtime 的构造入口。

CLI 通过 DefaultAgentFactory 创建 Runtime 和 Executor，
不直接接触 AgentLoopBuilder 和 RuntimeFactory。

Phase 3 过渡：返回旧 Protocol 的 Runtime 实例。
Phase 4+ 重写 Runtime 后，Factory 内部改为创建新 Protocol 实例。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcc_core import ToolContext
from mcc_core.runtime import AgentDefinition

from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor


@dataclass
class SessionContext:
    """Factory 创建 Runtime/Executor 所需的共享运行时依赖。

    封装 CLI 装配阶段构造的跨子系统共享对象，
    避免每个工厂方法都接收一长串参数。
    """

    llm: Any
    registry: Any
    tool_context: ToolContext
    work_dir: Path
    config: Any
    hooks: Any | None = None
    recorder: Any | None = None
    coordinator: Any | None = None
    agent_loop: Any | None = None
    teammate: Any | None = None
    bus: Any | None = None


@dataclass
class FactoryContext:
    """Factory 从 AgentDefinition 装配 Runtime/Executor 所需的共享基础设施。

    只含跨子系统共享的环境与基础设施对象，不含预构建的
    agent_loop/teammate——这些应由 Factory 从 AgentDefinition 装配。
    """

    llm: Any
    registry: Any
    tool_context: ToolContext
    work_dir: Path
    config: Any
    hooks: Any | None = None
    recorder: Any | None = None
    coordinator: Any | None = None
    bus: Any | None = None


class DefaultAgentFactory:
    """Agent 工厂：收拢 AgentLoop → AgentLoopExecutor → Runtime 的构造链。

    职责：
    - 从 AgentLoopBuilder 构建 AgentLoop 并包装为 InteractiveRuntimeV2。
    - 从 AgentLoop 创建 AgentLoopExecutor（Phase 2 适配层）。
    - 创建 PersistentRuntimeV2（委托 RuntimeFactory）。
    - 实现 mcc_core.runtime.protocols.AgentFactory Protocol（Phase 4+ 完整支持）。

    不负责配置解析或 Builder 组装——这些仍由 CLI wiring 层完成。
    Factory 只收拢"构造实例"这一步，使后续 Phase 4 重写 Runtime 时
    只需修改 Factory 内部实现，不需修改 CLI。
    """

    def __init__(self, session: SessionContext | None = None) -> None:
        self._session = session
        self._runtime_factory = _LazyRuntimeFactory()

    @property
    def session(self) -> SessionContext | None:
        return self._session

    def create_executor(
        self,
        definition_or_loop: Any,
        context: Any | None = None,
        *,
        runtime_id: str = "",
        agent_id: str = "",
    ) -> AgentLoopExecutor:
        """从 AgentDefinition 或已构造的 AgentLoop 创建 AgentLoopExecutor。

        新路径：create_executor(definition, context) 从 AgentDefinition
        的 profile 装配 AgentLoop 再包装为 AgentLoopExecutor。
        旧路径：create_executor(agent_loop, *, runtime_id, agent_id)
        直接包装已构造的 AgentLoop，保持向后兼容。
        """
        if isinstance(definition_or_loop, AgentDefinition):
            if context is None:
                raise ValueError(
                    "context is required when creating executor from AgentDefinition",
                )
            loop = self._assemble_agent_loop(definition_or_loop, context)
            return AgentLoopExecutor(
                loop,
                runtime_id=runtime_id,
                agent_id=agent_id or definition_or_loop.agent_id,
            )
        return AgentLoopExecutor(
            definition_or_loop,
            runtime_id=runtime_id,
            agent_id=agent_id,
        )

    @staticmethod
    def build_agent_loop(
        *,
        llm: Any,
        registry: Any,
        ctx: Any,
        system: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        max_iterations: int = 40,
        temperature: float = 0.0,
        permission: Any | None = None,
        max_cost_usd: float = 0.0,
        hooks: Any | None = None,
        permission_responses: Any | None = None,
        permission_timeout_secs: float = 120.0,
    ) -> Any:
        """统一 AgentLoop 构造入口（AR-001）。

        所有 AgentLoop 实例必须经此方法创建，禁止直接构造。
        """
        from mcc_harness.agent import AgentLoop

        return AgentLoop(
            llm=llm,
            registry=registry,
            ctx=ctx,
            system=system,
            model=model,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            temperature=temperature,
            permission=permission,
            max_cost_usd=max_cost_usd,
            hooks=hooks,
            permission_responses=permission_responses,
            permission_timeout_secs=permission_timeout_secs,
        )

    def build_interactive_runtime_v2(
        self,
        builder: Any,
        *,
        session_id: str = "",
        agent_id: str = "",
    ) -> Any:
        """从 AgentLoopBuilder 构建 AgentLoop 并包装为 InteractiveRuntimeV2。"""
        agent = builder.build()
        return self._runtime_factory.create_interactive_v2(
            agent,
            session_id=session_id,
            agent_id=agent_id,
        )

    def create_interactive_runtime(
        self,
        agent: Any,
        *,
        session_id: str = "",
        agent_id: str = "",
        runtime_id: str | None = None,
    ) -> Any:
        """从已构造的 AgentLoop 创建 InteractiveRuntimeV2。"""
        return self._runtime_factory.create_interactive_v2(
            agent,
            session_id=session_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
        )

    async def create(
        self,
        definition: AgentDefinition,
        *,
        session_context: object | None = None,
    ) -> Any:
        """实现 AgentFactory Protocol：从 AgentDefinition 创建 Runtime。

        根据 definition.runtime_kind 创建对应的 V2 Runtime：
        - INTERACTIVE → InteractiveRuntimeV2
        - TASK → TaskRuntime（通过 AgentLoopExecutor）
        - PERSISTENT → PersistentRuntimeV2（需 teammate + bus）
        - REMOTE → 暂不支持
        """
        from mcc_core.runtime.enums import RuntimeKind

        session = session_context or self._session
        if session is None:
            raise ValueError("session_context is required to create runtime")

        kind = definition.runtime_kind

        if kind is RuntimeKind.INTERACTIVE:
            from mcc_harness.runtime.interactive import InteractiveRuntimeV2

            agent_loop = getattr(session, "agent_loop", None)
            if agent_loop is None:
                raise ValueError(
                    "agent_loop is required in session_context for INTERACTIVE runtime",
                )
            return InteractiveRuntimeV2(
                agent_loop,
                agent_id=definition.agent_id,
            )

        if kind is RuntimeKind.TASK:
            from mcc_harness.runtime.task_runtime import TaskRuntime

            agent_loop = getattr(session, "agent_loop", None)
            if agent_loop is None:
                raise ValueError(
                    "agent_loop is required in session_context for TASK runtime",
                )
            executor = AgentLoopExecutor(
                agent_loop,
                agent_id=definition.agent_id,
            )
            return TaskRuntime(executor, agent_id=definition.agent_id)

        if kind is RuntimeKind.PERSISTENT:
            from mcc_harness.runtime.persistent import PersistentRuntimeV2

            teammate = getattr(session, "teammate", None)
            bus = getattr(session, "bus", None)
            if teammate is None or bus is None:
                raise ValueError(
                    "teammate and bus are required in session_context for PERSISTENT runtime",
                )
            return PersistentRuntimeV2.from_teammate(
                teammate,
                bus,
            )

        raise NotImplementedError(
            f"runtime_kind {kind} is not supported by create()",
        )

    async def create_runtime(
        self,
        definition: AgentDefinition,
        context: FactoryContext,
    ) -> Any:
        """从 AgentDefinition 的 profile 装配 Runtime。

        根据 definition.runtime_kind 从 AgentDefinition 装配
        AgentLoop → AgentLoopExecutor → Runtime：
        - INTERACTIVE → InteractiveRuntimeV2
        - TASK → TaskRuntime（通过 AgentLoopExecutor）
        - PERSISTENT → 需 teammate，暂不支持从 AgentDefinition 装配
        - REMOTE → 暂不支持
        """
        from mcc_core.runtime.enums import RuntimeKind

        kind = definition.runtime_kind

        if kind is RuntimeKind.INTERACTIVE:
            from mcc_harness.runtime.interactive import InteractiveRuntimeV2

            loop = self._assemble_agent_loop(definition, context)
            return InteractiveRuntimeV2(loop, agent_id=definition.agent_id)

        if kind is RuntimeKind.TASK:
            from mcc_harness.runtime.task_runtime import TaskRuntime

            loop = self._assemble_agent_loop(definition, context)
            executor = AgentLoopExecutor(loop, agent_id=definition.agent_id)
            return TaskRuntime(executor, agent_id=definition.agent_id)

        raise NotImplementedError(
            f"runtime_kind {kind} is not supported by create_runtime()"
            " from AgentDefinition; use create() with session_context"
            " for PERSISTENT runtime",
        )

    def _assemble_agent_loop(
        self,
        definition: AgentDefinition,
        context: FactoryContext,
    ) -> Any:
        """从 AgentDefinition 的 profile 装配 AgentLoop。

        model_profile → model/max_tokens/temperature；
        execution_profile → max_iterations；
        context.config → permission（若可用）。
        """
        if context.llm is None:
            raise ValueError(
                "llm is required in context to assemble agent loop",
            )
        permission = self._resolve_permission(context)
        return self.build_agent_loop(
            llm=context.llm,
            registry=context.registry,
            ctx=context.tool_context,
            model=definition.model_profile.model_id,
            max_tokens=definition.model_profile.max_tokens,
            max_iterations=definition.execution_profile.max_iterations,
            temperature=definition.model_profile.temperature,
            permission=permission,
            hooks=context.hooks,
        )

    @staticmethod
    def _resolve_permission(context: FactoryContext) -> Any | None:
        """从 FactoryContext.config 解析 PermissionChecker（若配置可用）。"""
        config = context.config
        if config is None:
            return None
        perms = getattr(config, "permissions", None)
        if perms is None:
            return None
        from mcc_harness import PermissionChecker

        security = getattr(config, "security", None)
        hardened = getattr(security, "hardened", True) if security else True
        return PermissionChecker(perms, hardened=hardened)


class _LazyRuntimeFactory:
    """延迟导入 RuntimeFactory，避免循环依赖。"""

    def __init__(self) -> None:
        self._factory: Any | None = None

    @property
    def _impl(self) -> Any:
        if self._factory is None:
            from mcc_harness.runtime.factory import RuntimeFactory

            self._factory = RuntimeFactory()
        return self._factory

    def create_interactive_v2(
        self,
        agent: Any,
        *,
        session_id: str = "",
        agent_id: str = "",
        runtime_id: str | None = None,
    ) -> Any:
        return self._impl.create_interactive_v2(
            agent,
            session_id=session_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
        )


__all__ = [
    "DefaultAgentFactory",
    "FactoryContext",
    "SessionContext",
]
