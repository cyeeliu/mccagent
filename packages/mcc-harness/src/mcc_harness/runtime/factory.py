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
        """创建 V2 常驻运行时（新 Protocol）。"""
        from .persistent import PersistentRuntimeV2

        return PersistentRuntimeV2.from_teammate(
            teammate,
            bus,
            lead_name=lead_name,
            runtime_id=runtime_id,
        )
