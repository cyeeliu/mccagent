"""统一 Runtime 抽象：为 Subagent、Teammate 与主 CLI Agent 提供一致的生命周期接口。

TaskRuntime 适配短生命周期子代理（单次执行后销毁），
PersistentRuntimeV2 适配常驻 teammate（循环执行直到 shutdown），
InteractiveRuntimeV2 适配 CLI 主 Agent（多轮交互式对话）。

核心协议 mcc_core.runtime.protocols.AgentRuntime 只包含通用生命周期操作。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mcc_core import Usage
from mcc_core.runtime.errors import (
    RuntimeStopTimeoutError,
    TaskTimeoutError,
    WaitTimeoutError,
)

__all__ = [
    "RuntimeMode",
    "RuntimeStatus",
    "RuntimeConfig",
    "RuntimeResult",
    "WaitTimeoutError",
    "RuntimeStopTimeoutError",
    "TaskTimeoutError",
]


class RuntimeMode(StrEnum):
    """运行时模式：短生命周期、常驻、交互式。"""

    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"
    INTERACTIVE = "interactive"


class RuntimeStatus(StrEnum):
    """统一生命周期状态。

    覆盖 Ephemeral（CREATED→RUNNING→终态）和
    Persistent（CREATED→STARTING→IDLE/RUNNING→…→SHUTDOWN）两种模式。
    IDLE/PAUSED 仅 Persistent 使用；BUDGET_EXCEEDED 仅 Ephemeral 使用。
    """

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    SHUTDOWN = "shutdown"


_TERMINAL_STATUSES = frozenset({
    RuntimeStatus.COMPLETED,
    RuntimeStatus.FAILED,
    RuntimeStatus.CANCELLED,
    RuntimeStatus.TIMEOUT,
    RuntimeStatus.BUDGET_EXCEEDED,
    RuntimeStatus.SHUTDOWN,
    "stopped",
})


@dataclass
class RuntimeConfig:
    """运行时配置：从 SubagentSpec / TeammateSpec / AgentLoop 提取的公共字段。

    模式差异通过 mode 字段表达，不堆叠布尔分支。
    require_plan / worktree_enabled 仅 Persistent 使用。
    agent_id / role / capabilities 用于 AgentRegistry 注册。
    session_id / parent_agent_id 用于关联会话和派生关系。
    metadata 承载扩展配置，不放入密钥或凭据。
    """

    name: str
    mode: RuntimeMode
    system: str = ""
    model: str = ""
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    max_tokens: int = 2048
    budget_tokens: int = 80_000
    timeout_secs: float = 120.0
    require_plan: bool = False
    worktree_enabled: bool = False
    agent_id: str = ""
    role: str = ""
    capabilities: list[str] = field(default_factory=list)
    session_id: str = ""
    parent_agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeResult:
    """统一运行结果：所有终态汇聚到同一结构（遗留兼容版本）。

    此版本使用 RuntimeStatus 枚举，供 V1 兼容路径（wait_task / wait_done）使用。
    新代码应优先使用 mcc_core.runtime.result.RuntimeResult（CoreRuntimeResult），
    它使用 TaskStatus 枚举并提供更强的类型安全。

    usage 为 None 表示用量未知（超时/取消/异常时无法获取），
    不伪造零消耗。
    agent_id / task_id 用于关联 Agent 和任务。
    artifacts 承载产出物标识（文件路径、URL 等）。
    """

    runtime_id: str
    status: RuntimeStatus
    summary: str
    usage: Usage | None = None
    cost_usd: float = 0.0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    agent_id: str = ""
    task_id: str | None = None
    artifacts: list[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES



def _terminal_result(
    runtime_id: str,
    status: RuntimeStatus,
    summary: str = "",
    *,
    usage: Usage | None = None,
    cost_usd: float = 0.0,
    error: str | None = None,
    started_at: float = 0.0,
    finished_at: float | None = None,
    agent_id: str = "",
    task_id: str | None = None,
    artifacts: list[str] | None = None,
) -> RuntimeResult:
    """构造终态结果的便捷工厂。"""
    return RuntimeResult(
        runtime_id=runtime_id,
        status=status,
        summary=summary,
        usage=usage,
        cost_usd=cost_usd,
        error=error,
        started_at=started_at,
        finished_at=finished_at if finished_at is not None else time.monotonic(),
        agent_id=agent_id,
        task_id=task_id,
        artifacts=artifacts if artifacts is not None else [],
    )
