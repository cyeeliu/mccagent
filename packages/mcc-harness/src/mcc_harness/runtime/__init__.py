"""统一 AgentRuntime 抽象。

为 Subagent（短生命周期）、Teammate（常驻）和 CLI 主 Agent（交互式）提供
一致的生命周期接口，通过 EphemeralRuntime、PersistentRuntime 和
InteractiveRuntime 适配。AgentStateMachine 提供独立于 Runtime 的状态管理。
"""

from __future__ import annotations

from .base import (
    AgentRuntime,
    InstructionInjectableRuntime,
    PausableRuntime,
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    WaitTimeoutError,
)
from .coordinator import RuntimeCoordinator, RuntimeSnapshot, ShutdownReport
from .ephemeral import EphemeralRuntime
from .factory import RuntimeFactory
from .interactive import InteractiveRuntime
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
    "InstructionInjectableRuntime",
    "InteractiveRuntime",
    "PausableRuntime",
    "PersistentRuntime",
    "RuntimeConfig",
    "RuntimeCoordinator",
    "RuntimeFactory",
    "RuntimeMode",
    "RuntimeResult",
    "RuntimeSnapshot",
    "RuntimeStatus",
    "ShutdownReport",
    "StateTransition",
    "TransitionGuard",
    "WaitTimeoutError",
]
