"""统一 AgentRuntime 抽象：为 Subagent、Teammate 与主 CLI Agent 提供一致的生命周期接口。

EphemeralRuntime 适配短生命周期子代理（单次执行后销毁），
PersistentRuntime 适配常驻 teammate（循环执行直到 shutdown），
InteractiveRuntime 适配 CLI 主 Agent（多轮交互式对话）。

核心协议 AgentRuntime 只包含通用生命周期操作（start/stop/cancel/wait_done）。
能力协议 PausableRuntime 和 InstructionInjectableRuntime 拆分出来，
调用方通过 isinstance 检查或 try/except 判断 Runtime 是否支持特定能力。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from mcc_core import Usage


class WaitTimeoutError(Exception):
    """等待超时：wait_done 的 timeout 到期，Runtime 可能仍在运行。

    与底层执行超时（RuntimeStatus.TIMEOUT）区分：
    - WaitTimeoutError 表示调用方等待超时，Runtime 未到达终态；
    - RuntimeStatus.TIMEOUT 表示 Runtime 内部执行超时，已到达终态。
    """


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
    """统一运行结果：所有终态汇聚到同一结构。

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


@runtime_checkable
class AgentRuntime(Protocol):
    """核心运行时协议：所有 Runtime 共同实现的生命周期接口。

    生命周期：start → (running | idle)* → stop/wait_done → 终态。
    cancel 为协作取消信号，在下一安全边界生效。
    pause/resume/inject_instruction 由 PausableRuntime 和
    InstructionInjectableRuntime 能力协议提供，不是所有 Runtime 都支持。
    """

    @property
    def runtime_id(self) -> str:
        """运行时唯一标识。"""
        ...

    @property
    def status(self) -> RuntimeStatus:
        """当前生命周期状态。"""
        ...

    @property
    def alive(self) -> bool:
        """是否仍在运行（未到达终态）。"""
        ...

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        """启动执行。Ephemeral 开始单次运行；Persistent 进入工作循环；
        Interactive 开始一个交互轮次。"""
        ...

    async def stop(self, *, timeout: float = 5.0) -> None:
        """请求停止并等待退出。"""
        ...

    def cancel(self) -> None:
        """发送取消信号（协作式，下一迭代边界生效）。"""
        ...

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        """等待运行时到达终态，返回统一结果。

        timeout 到期时抛出 WaitTimeoutError，不伪装成 RuntimeStatus.TIMEOUT。
        """
        ...


@runtime_checkable
class PausableRuntime(Protocol):
    """可暂停运行时协议：PersistentRuntime 实现，EphemeralRuntime 不支持。"""

    def pause(self) -> None:
        """暂停执行。"""
        ...

    def resume(self) -> None:
        """恢复执行。"""
        ...


@runtime_checkable
class InstructionInjectableRuntime(Protocol):
    """可注入指令运行时协议：PersistentRuntime 实现，EphemeralRuntime 不支持。"""

    def inject_instruction(self, instruction: str) -> None:
        """注入运行时指令（不打断当前轮，下一迭代边界生效）。"""
        ...


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
