"""Runtime 与 Task 状态枚举及合法迁移表。

Runtime 状态和 Task 状态分离（迁移方案 §3.6）：
- RuntimeStatus 描述执行实例生命周期
- TaskStatus 描述单次任务执行进度
"""

from __future__ import annotations

from enum import StrEnum


class RuntimeKind(StrEnum):
    INTERACTIVE = "interactive"
    TASK = "task"
    PERSISTENT = "persistent"
    REMOTE = "remote"


class RuntimeStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class TaskStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    DISPATCHED = "dispatched"
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RuntimeEventType(StrEnum):
    RUNTIME_STARTED = "runtime_started"
    RUNTIME_READY = "runtime_ready"
    RUNTIME_STOPPING = "runtime_stopping"
    RUNTIME_STOPPED = "runtime_stopped"
    RUNTIME_FAILED = "runtime_failed"
    TASK_ACCEPTED = "task_accepted"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_TIMED_OUT = "task_timed_out"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"
    ARTIFACT_CREATED = "artifact_created"
    HEALTH_CHANGED = "health_changed"


_RUNTIME_TRANSITIONS: dict[RuntimeStatus, frozenset[RuntimeStatus]] = {
    RuntimeStatus.CREATED: frozenset({RuntimeStatus.STARTING, RuntimeStatus.FAILED}),
    RuntimeStatus.STARTING: frozenset({RuntimeStatus.READY, RuntimeStatus.FAILED}),
    RuntimeStatus.READY: frozenset({
        RuntimeStatus.PAUSING,
        RuntimeStatus.STOPPING,
        RuntimeStatus.FAILED,
    }),
    RuntimeStatus.PAUSING: frozenset({
        RuntimeStatus.PAUSED,
        RuntimeStatus.READY,
        RuntimeStatus.FAILED,
    }),
    RuntimeStatus.PAUSED: frozenset({
        RuntimeStatus.READY,
        RuntimeStatus.STOPPING,
        RuntimeStatus.FAILED,
    }),
    RuntimeStatus.STOPPING: frozenset({RuntimeStatus.STOPPED, RuntimeStatus.FAILED}),
    RuntimeStatus.STOPPED: frozenset(),
    RuntimeStatus.FAILED: frozenset(),
}

_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.QUEUED, TaskStatus.DISPATCHED, TaskStatus.CANCELLED}),
    TaskStatus.QUEUED: frozenset({TaskStatus.DISPATCHED, TaskStatus.CANCELLED}),
    TaskStatus.DISPATCHED: frozenset({
        TaskStatus.ACCEPTED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.ACCEPTED: frozenset({
        TaskStatus.RUNNING,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.DISPATCHED,
    }),
    TaskStatus.RUNNING: frozenset({
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.TIMED_OUT,
        TaskStatus.DISPATCHED,
    }),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset({TaskStatus.DISPATCHED}),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.TIMED_OUT: frozenset({TaskStatus.DISPATCHED}),
}

RUNTIME_TERMINAL_STATUSES = frozenset({RuntimeStatus.STOPPED, RuntimeStatus.FAILED})
TASK_TERMINAL_STATUSES = frozenset({
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.TIMED_OUT,
})


def is_valid_runtime_transition(source: RuntimeStatus, target: RuntimeStatus) -> bool:
    return target in _RUNTIME_TRANSITIONS.get(source, frozenset())


def is_valid_task_transition(source: TaskStatus, target: TaskStatus) -> bool:
    return target in _TASK_TRANSITIONS.get(source, frozenset())


def is_runtime_terminal(status: RuntimeStatus) -> bool:
    return status in RUNTIME_TERMINAL_STATUSES


def is_task_terminal(status: TaskStatus) -> bool:
    return status in TASK_TERMINAL_STATUSES


__all__ = [
    "RuntimeKind",
    "RuntimeStatus",
    "TaskStatus",
    "RuntimeEventType",
    "RUNTIME_TERMINAL_STATUSES",
    "TASK_TERMINAL_STATUSES",
    "is_valid_runtime_transition",
    "is_valid_task_transition",
    "is_runtime_terminal",
    "is_task_terminal",
]
