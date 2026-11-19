"""RuntimeFactory：运行时工厂，集中创建各类 Runtime。

负责构造 V2 Runtime（InteractiveRuntimeV2、TaskRuntime、PersistentRuntimeV2），
不负责规划任务或选择 Agent。runtime_id 使用 UUID 避免冲突。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..teammate import LEAD_NAME, TeammateRuntime


def _generate_runtime_id(prefix: str) -> str:
    """生成带前缀的 UUID 运行时 ID。"""
    return f"{prefix}::{uuid4().hex[:12]}"


class RuntimeFactory:
    """运行时工厂：集中创建各类 Runtime 实例。

    每次调用生成新的 Runtime 实例和唯一 runtime_id。
    不缓存或复用实例，不负责注册到 Coordinator。
    """

    def create_interactive_v2(
        self,
        agent: Any,
        *,
        session_id: str = "",
        agent_id: str = "",
        runtime_id: str | None = None,
    ) -> Any:
        """创建 V2 交互式运行时（新 Protocol）。"""
        from .interactive import InteractiveRuntimeV2

        return InteractiveRuntimeV2(
            agent,
            runtime_id=runtime_id,
            session_id=session_id,
            agent_id=agent_id,
        )

    def create_task(
        self,
        executor: Any,
        *,
        agent_id: str = "",
        runtime_id: str | None = None,
    ) -> Any:
        """创建 V2 任务运行时（新 Protocol）。"""
        from .task_runtime import TaskRuntime

        return TaskRuntime(
            executor,
            runtime_id=runtime_id,
            agent_id=agent_id,
        )

    def create_persistent_v2(
        self,
        teammate: TeammateRuntime,
        bus: Any,
        *,
        lead_name: str = LEAD_NAME,
        runtime_id: str | None = None,
    ) -> Any:
        """创建 V2 常驻运行时（新 Protocol）。

        通过 teammate.build_agent_executor() 构造 AgentLoopExecutor，
        包装为 TeammateExecutionService 后用 from_executor 创建实例。
        teammate 仅负责消息收发，不执行任务。
        """
        from ..teammate import (
            TeammateDefinition,
            TeammateExecutionService,
            TeammateMessageAdapter,
        )
        from .persistent import PersistentRuntimeV2

        inner_executor = teammate.build_agent_executor()
        definition = TeammateDefinition.from_spec(teammate.spec)
        message_adapter = TeammateMessageAdapter(
            bus,
            name=teammate.spec.name,
            lead_name=lead_name,
        )
        service = TeammateExecutionService(
            inner_executor,
            definition=definition,
            message_adapter=message_adapter,
            plan_store=teammate._plan_store,
            task_store=teammate._task_store,
            base_dir=teammate._base_dir,
            worktree_enabled=teammate._worktree_enabled,
            runtime_id=teammate.runtime_id,
            agent_id=teammate.spec.name,
        )
        return PersistentRuntimeV2.from_executor(
            service,
            teammate=teammate,
            bus=bus,
            lead_name=lead_name,
            runtime_id=runtime_id,
            agent_id=teammate.spec.name,
        )
