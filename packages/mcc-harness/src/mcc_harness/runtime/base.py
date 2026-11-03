"""统一 AgentRuntime 抽象：为 Subagent 与 Teammate 提供一致的生命周期接口。

EphemeralRuntime 适配短生命周期子代理（单次执行后销毁），
PersistentRuntime 适配常驻 teammate（循环执行直到 shutdown）。
两者通过同一 AgentRuntime Protocol 暴露 start/stop/cancel/pause/resume/
inject_instruction/wait_done 方法，上层编排代码无需区分运行时模式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from mcc_core import Usage


class RuntimeMode(StrEnum):
    """运行时模式：短生命周期 vs 常驻。"""

    EPHEMERAL = "ephemeral"
    PERSISTENT = "persistent"


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
    """运行时配置：从 SubagentSpec / TeammateSpec 提取的公共字段。

    模式差异通过 mode 字段表达，不堆叠布尔分支。
    require_plan / worktree_enabled 仅 Persistent 使用。
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


@dataclass
class RuntimeResult:
    """统一运行结果：所有终态汇聚到同一结构。

    usage 为 None 表示用量未知（超时/取消/异常时无法获取），
    不伪造零消耗。
    """

    runtime_id: str
    status: RuntimeStatus
    summary: str
    usage: Usage | None = None
    cost_usd: float = 0.0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@runtime_checkable
class AgentRuntime(Protocol):
    """统一运行时协议：Ephemeral 与 Persistent 共同实现。

    生命周期：start → (running | idle | paused)* → stop/wait_done → 终态。
    控制信号：cancel（协作取消）、pause/resume（仅 Persistent）、
    inject_instruction（运行时注入，不打断当前轮）。
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

    async def start(self, task: str, *, task_id: str | None = None) -> None:
        """启动执行。Ephemeral 开始单次运行；Persistent 进入工作循环。"""
        ...

    async def stop(self, *, timeout: float = 5.0) -> None:
        """请求停止并等待退出。"""
        ...

    def cancel(self) -> None:
        """发送取消信号（协作式，下一迭代边界生效）。"""
        ...

    def pause(self) -> None:
        """暂停执行（仅 Persistent 支持）。"""
        ...

    def resume(self) -> None:
        """恢复执行（仅 Persistent 支持）。"""
        ...

    def inject_instruction(self, instruction: str) -> None:
        """注入运行时指令（不打断当前轮，下一迭代边界生效）。"""
        ...

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        """等待运行时到达终态，返回统一结果。"""
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
    )
