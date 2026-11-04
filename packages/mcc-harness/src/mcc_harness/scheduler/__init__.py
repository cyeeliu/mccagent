"""任务调度器模块：依赖检测、优先级排序、能力匹配、负载约束与重试。

提供可插拔策略接口，实现 TaskScheduler Protocol，
可直接注入 TeamOrchestrator 替换默认 CapabilityScheduler。
"""

from __future__ import annotations

from .base import (
    AgentCandidate,
    ConcurrencyExceededError,
    NoAgentAvailableError,
    PriorityOrder,
    RetryExhaustedError,
    RetryPolicy,
    SchedulingError,
    SchedulingPolicy,
    SchedulingResult,
    TaskCandidate,
    TaskNotReadyError,
)
from .scheduler import LeaseManager, RetryHandler, TaskSchedulerImpl

__all__ = [
    "AgentCandidate",
    "ConcurrencyExceededError",
    "LeaseManager",
    "NoAgentAvailableError",
    "PriorityOrder",
    "RetryExhaustedError",
    "RetryHandler",
    "RetryPolicy",
    "SchedulingError",
    "SchedulingPolicy",
    "SchedulingResult",
    "TaskCandidate",
    "TaskNotReadyError",
    "TaskSchedulerImpl",
]
