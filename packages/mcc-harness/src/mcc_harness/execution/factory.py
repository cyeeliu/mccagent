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
        agent_loop: Any,
        *,
        runtime_id: str = "",
        agent_id: str = "",
    ) -> AgentLoopExecutor:
        """从已构造的 AgentLoop 创建 AgentLoopExecutor。"""
        return AgentLoopExecutor(
            agent_loop,
            runtime_id=runtime_id,
            agent_id=agent_id,
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
                    "teammate and bus are required in session_context"
                    " for PERSISTENT runtime",
                )
            return PersistentRuntimeV2.from_teammate(
                teammate,
                bus,
            )

        raise NotImplementedError(
            f"runtime_kind {kind} is not supported by create()",
        )


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
    "SessionContext",
]
