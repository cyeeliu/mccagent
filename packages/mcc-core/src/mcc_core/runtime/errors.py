"""Runtime 领域错误类型。

所有错误继承 DomainError，携带稳定 error_code 供 RetryPolicy 匹配。
"""

from __future__ import annotations


class DomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


class InvalidTransitionError(DomainError):
    def __init__(self, source: str, target: str) -> None:
        super().__init__(
            code="invalid_transition",
            message=f"非法状态迁移: {source} -> {target}",
            retryable=False,
            details={"source": source, "target": target},
        )


class RuntimeNotReadyError(DomainError):
    def __init__(self, runtime_id: str, status: str) -> None:
        super().__init__(
            code="runtime_not_ready",
            message=f"Runtime {runtime_id} 未就绪 (当前状态: {status})",
            retryable=True,
            details={"runtime_id": runtime_id, "status": status},
        )


class TaskValidationError(DomainError):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(
            code="task_validation_error",
            message=message,
            retryable=False,
            details=details,
        )


class IdempotencyConflictError(DomainError):
    def __init__(self, idempotency_key: str) -> None:
        super().__init__(
            code="idempotency_conflict",
            message=f"幂等键冲突: {idempotency_key}",
            retryable=False,
            details={"idempotency_key": idempotency_key},
        )


class UnsupportedOperationError(DomainError):
    def __init__(self, operation: str, runtime_id: str) -> None:
        super().__init__(
            code="unsupported_operation",
            message=f"Runtime {runtime_id} 不支持操作: {operation}",
            retryable=False,
            details={"operation": operation, "runtime_id": runtime_id},
        )


class RuntimeNotFoundError(DomainError):
    def __init__(self, runtime_id: str) -> None:
        super().__init__(
            code="runtime_not_found",
            message=f"Runtime 不存在: {runtime_id}",
            retryable=False,
            details={"runtime_id": runtime_id},
        )


class TaskTimeoutError(DomainError):
    def __init__(self, task_id: str, attempt_id: str) -> None:
        super().__init__(
            code="task_timeout",
            message=f"任务执行超时: task={task_id}, attempt={attempt_id}",
            retryable=True,
            details={"task_id": task_id, "attempt_id": attempt_id},
        )


class WaitTimeoutError(DomainError):
    """等待超时：调用方等待 Runtime 结果时超时，Runtime 可能仍在运行。

    与 TaskTimeoutError 区分：
    - WaitTimeoutError 表示调用方等待超时，Runtime 未到达终态；
    - TaskTimeoutError 表示 Runtime 内部执行超时，已到达终态。
    """

    def __init__(self, runtime_id: str, timeout: float) -> None:
        super().__init__(
            code="wait_timeout",
            message=f"等待超时: runtime={runtime_id}, timeout={timeout}s",
            retryable=True,
            details={"runtime_id": runtime_id, "timeout": timeout},
        )


class RuntimeStopTimeoutError(DomainError):
    """Runtime 停止超时：stop 操作在指定超时内未能完成。

    与 WaitTimeoutError 区分：
    - RuntimeStopTimeoutError 专指 stop/shutdown 操作超时；
    - WaitTimeoutError 专指等待任务结果超时。
    """

    def __init__(self, runtime_id: str, timeout: float) -> None:
        super().__init__(
            code="runtime_stop_timeout",
            message=f"Runtime 停止超时: runtime={runtime_id}, timeout={timeout}s",
            retryable=False,
            details={"runtime_id": runtime_id, "timeout": timeout},
        )


class TaskNotFoundError(DomainError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            code="task_not_found",
            message=f"任务不存在: {task_id}",
            retryable=False,
            details={"task_id": task_id},
        )


class AttemptNotFoundError(DomainError):
    def __init__(self, attempt_id: str) -> None:
        super().__init__(
            code="attempt_not_found",
            message=f"执行尝试不存在: {attempt_id}",
            retryable=False,
            details={"attempt_id": attempt_id},
        )


class LeaseExpiredError(DomainError):
    def __init__(self, task_id: str) -> None:
        super().__init__(
            code="lease_expired",
            message=f"任务租约已过期: {task_id}",
            retryable=True,
            details={"task_id": task_id},
        )


class CASConflictError(DomainError):
    def __init__(self, task_id: str, expected: int, actual: int) -> None:
        super().__init__(
            code="cas_conflict",
            message=f"CAS 冲突: task={task_id}, expected version={expected}, actual={actual}",
            retryable=True,
            details={"task_id": task_id, "expected": expected, "actual": actual},
        )


__all__ = [
    "DomainError",
    "InvalidTransitionError",
    "RuntimeNotReadyError",
    "TaskValidationError",
    "IdempotencyConflictError",
    "UnsupportedOperationError",
    "RuntimeNotFoundError",
    "TaskTimeoutError",
    "WaitTimeoutError",
    "RuntimeStopTimeoutError",
    "TaskNotFoundError",
    "AttemptNotFoundError",
    "LeaseExpiredError",
    "CASConflictError",
]
