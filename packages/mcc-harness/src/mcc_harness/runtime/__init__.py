"""统一 AgentRuntime 抽象。

为 Subagent（短生命周期）、Teammate（常驻）和 CLI 主 Agent（交互式）提供
一致的生命周期接口，通过 PersistentRuntimeV2 和 InteractiveRuntimeV2 适配。
AgentStateMachine 提供独立于 Runtime 的状态管理。
"""

from __future__ import annotations

from mcc_core.runtime.result import RuntimeResult as CoreRuntimeResult

from .base import (
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    RuntimeStopTimeoutError,
    TaskTimeoutError,
    WaitTimeoutError,
)
from .bounded_task_queue import (
    BoundedTaskQueue,
    QueueClosedError,
    QueueFullError,
)
from .coordinator import (
    CoordinatorClosedError,
    RuntimeCoordinator,
    RuntimeSnapshot,
    ShutdownReport,
)
from .dispatcher import DispatchError, TaskDispatcher
from .factory import RuntimeFactory
from .interactive import InteractiveRuntimeV2
from .persistent import PersistentRuntimeV2
from .state_machine import (
    AgentState,
    AgentStateMachine,
    CallbackGuard,
    GuardRejectedError,
    IllegalTransitionError,
    StateTransition,
    TransitionGuard,
)
from .task_store_adapter import TaskStoreAdapter
from .worker_pool import WorkerPool

__all__ = [
    "AgentState",
    "AgentStateMachine",
    "BoundedTaskQueue",
    "CallbackGuard",
    "CoordinatorClosedError",
    "CoreRuntimeResult",
    "DispatchError",
    "GuardRejectedError",
    "IllegalTransitionError",
    "InteractiveRuntimeV2",
    "PersistentRuntimeV2",
    "QueueClosedError",
    "QueueFullError",
    "RuntimeConfig",
    "RuntimeCoordinator",
    "RuntimeFactory",
    "RuntimeMode",
    "RuntimeResult",
    "RuntimeSnapshot",
    "RuntimeStatus",
    "RuntimeStopTimeoutError",
    "ShutdownReport",
    "StateTransition",
    "TaskDispatcher",
    "TaskStoreAdapter",
    "TaskTimeoutError",
    "TransitionGuard",
    "WaitTimeoutError",
    "WorkerPool",
]
