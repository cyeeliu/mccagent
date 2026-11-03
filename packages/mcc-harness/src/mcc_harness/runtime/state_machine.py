"""Agent 生命周期状态机：独立于 Runtime 实现的可复用状态管理。

将 Agent 状态迁移逻辑从具体 Runtime（Ephemeral/Persistent）中解耦，
形成可独立单测的状态机组件。Runtime、Scheduler 和监控读取同一状态源。

与 TaskStateMachine（mcc_tools.task）关联但职责不同：
- AgentStateMachine 管理 Agent 运行时生命周期（created→running→completed）
- TaskStateMachine 管理任务记录生命周期（pending→in_progress→completed）
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from .base import RuntimeMode, RuntimeStatus

AgentState = RuntimeStatus

_TERMINAL_STATES: frozenset[AgentState] = frozenset({
    AgentState.COMPLETED,
    AgentState.FAILED,
    AgentState.CANCELLED,
    AgentState.TIMEOUT,
    AgentState.BUDGET_EXCEEDED,
    AgentState.SHUTDOWN,
})

_EPHEMERAL_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.CREATED: frozenset({
        AgentState.RUNNING,
        AgentState.CANCELLED,
        AgentState.FAILED,
    }),
    AgentState.RUNNING: frozenset({
        AgentState.PAUSED,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMEOUT,
        AgentState.BUDGET_EXCEEDED,
    }),
    AgentState.PAUSED: frozenset({
        AgentState.RUNNING,
        AgentState.CANCELLED,
    }),
    AgentState.COMPLETED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.CANCELLED: frozenset(),
    AgentState.TIMEOUT: frozenset(),
    AgentState.BUDGET_EXCEEDED: frozenset(),
}

_PERSISTENT_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.CREATED: frozenset({
        AgentState.STARTING,
        AgentState.SHUTDOWN,
    }),
    AgentState.STARTING: frozenset({
        AgentState.IDLE,
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.SHUTDOWN,
    }),
    AgentState.IDLE: frozenset({
        AgentState.RUNNING,
        AgentState.SHUTDOWN,
    }),
    AgentState.RUNNING: frozenset({
        AgentState.IDLE,
        AgentState.PAUSED,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMEOUT,
        AgentState.BUDGET_EXCEEDED,
        AgentState.STOPPING,
    }),
    AgentState.PAUSED: frozenset({
        AgentState.RUNNING,
        AgentState.STOPPING,
        AgentState.SHUTDOWN,
    }),
    AgentState.COMPLETED: frozenset({
        AgentState.IDLE,
        AgentState.STOPPING,
    }),
    AgentState.FAILED: frozenset({
        AgentState.IDLE,
        AgentState.STOPPING,
    }),
    AgentState.CANCELLED: frozenset({
        AgentState.IDLE,
        AgentState.STOPPING,
    }),
    AgentState.TIMEOUT: frozenset({
        AgentState.IDLE,
        AgentState.STOPPING,
    }),
    AgentState.BUDGET_EXCEEDED: frozenset({
        AgentState.STOPPING,
    }),
    AgentState.STOPPING: frozenset({
        AgentState.SHUTDOWN,
    }),
    AgentState.SHUTDOWN: frozenset(),
}

_DEFAULT_TRANSITIONS: dict[RuntimeMode, dict[AgentState, frozenset[AgentState]]] = {
    RuntimeMode.EPHEMERAL: _EPHEMERAL_TRANSITIONS,
    RuntimeMode.PERSISTENT: _PERSISTENT_TRANSITIONS,
}


class IllegalTransitionError(RuntimeError):
    """状态迁移非法：当前状态不允许迁移到目标状态。"""


class GuardRejectedError(RuntimeError):
    """迁移守卫拒绝：守卫条件未满足。"""


@dataclass(frozen=True)
class StateTransition:
    """单次状态迁移审计记录。

    记录来源、目标、原因、时间戳、操作者和追踪 ID，
    供监控、调试和审计使用。
    """

    from_state: AgentState
    to_state: AgentState
    reason: str = ""
    timestamp: float = 0.0
    actor: str = ""
    trace_id: str = ""


class TransitionGuard(Protocol):
    """迁移守卫：在状态变更前检查是否允许。

    返回 True 允许迁移，False 拒绝。
    守卫不修改状态，只做只读检查。
    """

    def check(self, current: AgentState, target: AgentState) -> bool: ...


@dataclass(frozen=True)
class CallbackGuard:
    """用回调函数实现的迁移守卫。"""

    predicate: Callable[[AgentState, AgentState], bool]
    description: str = ""

    def check(self, current: AgentState, target: AgentState) -> bool:
        return self.predicate(current, target)


class AgentStateMachine:
    """Agent 生命周期状态机：独立于 Runtime 实现的可复用状态管理。

    职责：
    - 维护当前状态和迁移历史
    - 校验迁移合法性（迁移表 + 守卫）
    - 通知状态变更订阅者
    - 记录审计信息（reason/timestamp/actor/trace_id）

    不执行任何 Agent 业务逻辑。Runtime、Scheduler 和监控
    通过此状态机读取和变更 Agent 生命周期状态。
    """

    def __init__(
        self,
        agent_id: str,
        *,
        initial_state: AgentState = AgentState.CREATED,
        mode: RuntimeMode = RuntimeMode.EPHEMERAL,
        transitions: dict[AgentState, frozenset[AgentState]] | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._state = initial_state
        self._mode = mode
        self._transitions = (
            transitions if transitions is not None else _DEFAULT_TRANSITIONS[mode]
        )
        self._history: list[StateTransition] = []
        self._subscribers: list[Callable[[StateTransition], None]] = []
        self._guards: dict[tuple[AgentState, AgentState], list[TransitionGuard]] = {}
        self._lock = threading.Lock()

    @property
    def agent_id(self) -> str:
        """Agent 唯一标识。"""
        return self._agent_id

    @property
    def state(self) -> AgentState:
        """当前生命周期状态。"""
        return self._state

    @property
    def mode(self) -> RuntimeMode:
        """运行时模式。"""
        return self._mode

    @property
    def is_terminal(self) -> bool:
        """是否已到达终态。"""
        return self._state in _TERMINAL_STATES

    @property
    def history(self) -> list[StateTransition]:
        """迁移历史（只读副本）。"""
        return list(self._history)

    @property
    def last_transition(self) -> StateTransition | None:
        """最近一次迁移记录。"""
        return self._history[-1] if self._history else None

    def can_transition(self, target: AgentState) -> bool:
        """检查当前状态是否允许迁移到目标状态（不考虑守卫）。"""
        return target in self._transitions.get(self._state, frozenset())

    def allowed_targets(self) -> frozenset[AgentState]:
        """当前状态允许迁移到的目标状态集合。"""
        return self._transitions.get(self._state, frozenset())

    def transition(
        self,
        target: AgentState,
        *,
        reason: str = "",
        actor: str = "",
        trace_id: str = "",
    ) -> StateTransition:
        """执行状态迁移（唯一入口）。

        校验迁移表和守卫，通过后更新状态、记录审计、通知订阅者。
        非法迁移抛出 IllegalTransitionError，守卫拒绝抛出 GuardRejectedError。

        Args:
            target: 目标状态
            reason: 迁移原因（人类可读）
            actor: 触发迁移的操作者（agent_id 或 system）
            trace_id: 追踪 ID（用于关联请求链路）

        Returns:
            迁移审计记录

        Raises:
            IllegalTransitionError: 当前状态不允许迁移到目标状态
            GuardRejectedError: 守卫条件未满足
        """
        with self._lock:
            current = self._state
            if not self.can_transition(target):
                raise IllegalTransitionError(
                    f"agent {self._agent_id} cannot transition from "
                    f"{current.value} to {target.value}"
                )
            for guard in self._guards.get((current, target), []):
                if not guard.check(current, target):
                    raise GuardRejectedError(
                        f"guard rejected transition {current.value}"
                        f" -> {target.value} for agent {self._agent_id}"
                    )
            record = StateTransition(
                from_state=current,
                to_state=target,
                reason=reason,
                timestamp=time.monotonic(),
                actor=actor,
                trace_id=trace_id,
            )
            self._state = target
            self._history.append(record)
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            subscriber(record)
        return record

    def subscribe(
        self,
        callback: Callable[[StateTransition], None],
    ) -> Callable[[], None]:
        """订阅状态变更事件。

        每次成功迁移后调用 callback(record)。
        返回取消订阅函数。
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock, suppress(ValueError):
                self._subscribers.remove(callback)

        return unsubscribe

    def add_guard(
        self,
        from_state: AgentState,
        to_state: AgentState,
        guard: TransitionGuard,
    ) -> None:
        """为特定迁移添加守卫。

        守卫在迁移表校验通过后、状态变更前执行。
        多个守卫全部通过才允许迁移。
        """
        with self._lock:
            key = (from_state, to_state)
            self._guards.setdefault(key, []).append(guard)
