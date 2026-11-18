"""Runtime 领域模型与统一协议。

此包是 AgentRuntime 架构的核心领域层，不依赖 mcc-harness / mcc-cli / mcc-tools。
所有 Runtime 实现共享此包定义的 AgentRuntime Protocol 和领域类型。

迁移方案 Phase 1 交付物。
"""

from __future__ import annotations

from mcc_core.runtime.capabilities import (
    INTERACTIVE_CAPABILITIES,
    PERSISTENT_CAPABILITIES,
    TASK_CAPABILITIES,
    RuntimeCapabilities,
)
from mcc_core.runtime.definition import (
    AgentDefinition,
    ExecutionProfile,
    ModelProfile,
    PermissionProfile,
    ToolProfile,
)
from mcc_core.runtime.enums import (
    RUNTIME_TERMINAL_STATUSES,
    TASK_TERMINAL_STATUSES,
    RuntimeEventType,
    RuntimeKind,
    RuntimeStatus,
    TaskStatus,
    is_runtime_terminal,
    is_task_terminal,
    is_valid_runtime_transition,
    is_valid_task_transition,
)
from mcc_core.runtime.errors import (
    AttemptNotFoundError,
    CASConflictError,
    DomainError,
    IdempotencyConflictError,
    InvalidTransitionError,
    LeaseExpiredError,
    RuntimeNotFoundError,
    RuntimeNotReadyError,
    TaskNotFoundError,
    TaskTimeoutError,
    TaskValidationError,
    UnsupportedOperationError,
)
from mcc_core.runtime.event import RuntimeEvent
from mcc_core.runtime.handle import TaskHandle
from mcc_core.runtime.health import HealthState, RuntimeHealth
from mcc_core.runtime.protocols import (
    AgentExecutor,
    AgentFactory,
    AgentRuntime,
    CancellationToken,
    EventEmitter,
)
from mcc_core.runtime.result import (
    ArtifactRef,
    RuntimeErrorInfo,
    RuntimeResult,
    TokenUsage,
)
from mcc_core.runtime.task import RuntimeTask, TaskContext
from mcc_core.runtime.task_record import (
    ATTEMPT_TERMINAL_STATUSES,
    AssignmentStatus,
    AttemptStatus,
    TaskAssignment,
    TaskAttempt,
    TaskRecord,
)
from mcc_core.runtime.task_store import InMemoryTaskStore, TaskStoreProtocol

__all__ = [
    "AgentDefinition",
    "AgentExecutor",
    "AgentFactory",
    "AgentRuntime",
    "ArtifactRef",
    "AttemptNotFoundError",
    "AttemptStatus",
    "AssignmentStatus",
    "ATTEMPT_TERMINAL_STATUSES",
    "CancellationToken",
    "CASConflictError",
    "DomainError",
    "EventEmitter",
    "ExecutionProfile",
    "HealthState",
    "IdempotencyConflictError",
    "InMemoryTaskStore",
    "INTERACTIVE_CAPABILITIES",
    "InvalidTransitionError",
    "LeaseExpiredError",
    "ModelProfile",
    "PermissionProfile",
    "PERSISTENT_CAPABILITIES",
    "RuntimeCapabilities",
    "RuntimeErrorInfo",
    "RuntimeEvent",
    "RuntimeEventType",
    "RuntimeHealth",
    "RuntimeKind",
    "RuntimeNotFoundError",
    "RuntimeNotReadyError",
    "RuntimeResult",
    "RuntimeStatus",
    "RUNTIME_TERMINAL_STATUSES",
    "TASK_CAPABILITIES",
    "TaskAssignment",
    "TaskAttempt",
    "TaskContext",
    "TaskHandle",
    "TaskNotFoundError",
    "TaskRecord",
    "TaskStatus",
    "TASK_TERMINAL_STATUSES",
    "TaskStoreProtocol",
    "TaskTimeoutError",
    "TaskValidationError",
    "TokenUsage",
    "ToolProfile",
    "UnsupportedOperationError",
    "is_runtime_terminal",
    "is_task_terminal",
    "is_valid_runtime_transition",
    "is_valid_task_transition",
    "RuntimeTask",
]
