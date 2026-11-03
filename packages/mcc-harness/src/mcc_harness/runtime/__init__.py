"""统一 AgentRuntime 抽象。

为 Subagent（短生命周期）和 Teammate（常驻）提供一致的
生命周期接口，通过 EphemeralRuntime 和 PersistentRuntime 适配。
AgentStateMachine 提供独立于 Runtime 的状态管理。
"""

from __future__ import annotations

from .base import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
)
from .ephemeral import EphemeralRuntime
from .persistent import PersistentRuntime
from .state_machine import (
    AgentState,
    AgentStateMachine,
    CallbackGuard,
    GuardRejectedError,
    IllegalTransitionError,
    StateTransition,
    TransitionGuard,
)

__all__ = [
    "AgentRuntime",
    "AgentState",
    "AgentStateMachine",
    "CallbackGuard",
    "EphemeralRuntime",
    "GuardRejectedError",
    "IllegalTransitionError",
    "PersistentRuntime",
    "RuntimeConfig",
    "RuntimeMode",
    "RuntimeResult",
    "RuntimeStatus",
    "StateTransition",
    "TransitionGuard",
]
