"""RuntimeFactory：运行时工厂，集中创建各类 Runtime。

负责构造 InteractiveRuntime、EphemeralRuntime 和 PersistentRuntime，
不负责规划任务或选择 Agent。runtime_id 使用 UUID 避免冲突。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from mcc_core import ToolContext

from ..subagent import SubagentRunner, SubagentSpec
from ..teammate import LEAD_NAME, TeammateRuntime
from .ephemeral import EphemeralRuntime
from .interactive import InteractiveRuntime
from .persistent import PersistentRuntime


def _generate_runtime_id(prefix: str) -> str:
    """生成带前缀的 UUID 运行时 ID。"""
    return f"{prefix}::{uuid4().hex[:12]}"


class RuntimeFactory:
    """运行时工厂：集中创建各类 Runtime 实例。

    每次调用生成新的 Runtime 实例和唯一 runtime_id。
    不缓存或复用实例，不负责注册到 Coordinator。
    """

    def create_interactive(
        self,
        agent: Any,
        *,
        session_id: str = "",
        agent_id: str = "",
        runtime_id: str | None = None,
    ) -> InteractiveRuntime:
        """创建交互式运行时（CLI 主 Agent）。"""
        return InteractiveRuntime(
            agent,
            runtime_id=runtime_id,
            session_id=session_id,
            agent_id=agent_id,
        )

    def create_ephemeral(
        self,
        runner: SubagentRunner,
        spec: SubagentSpec,
        ctx: ToolContext,
        *,
        channel: Any = None,
    ) -> EphemeralRuntime:
        """创建短生命周期运行时（Subagent）。"""
        return EphemeralRuntime(runner, spec, ctx, channel=channel)

    def create_persistent(
        self,
        teammate: TeammateRuntime,
        bus: Any,
        *,
        lead_name: str = LEAD_NAME,
    ) -> PersistentRuntime:
        """创建常驻运行时（Teammate）。"""
        return PersistentRuntime(teammate, bus, lead_name=lead_name)
