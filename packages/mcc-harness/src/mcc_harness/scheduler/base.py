"""任务调度器类型定义：候选对象、策略与结果。

将调度决策所需的结构化输入输出集中定义，
与 AgentRegistry、TaskStore 解耦。
调度器通过 TaskCandidate 描述待调度任务，
通过 AgentCandidate 描述可用 Agent，
依据 SchedulingPolicy 做分配决策，产出 SchedulingResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mcc_harness.team.orchestrator import Assignment, TaskSpec


class SchedulingError(RuntimeError):
    """调度器基础错误。"""


class TaskNotReadyError(SchedulingError):
    """任务依赖未满足，不可调度。"""


class NoAgentAvailableError(SchedulingError):
    """无可用 Agent 匹配任务能力要求。"""


class ConcurrencyExceededError(SchedulingError):
    """并发上限已达，无法分配。"""


class RetryExhaustedError(SchedulingError):
    """任务重试次数耗尽。"""


class PriorityOrder(StrEnum):
    """优先级排序方向。"""

    DESC = "desc"
    ASC = "asc"


@dataclass(frozen=True)
class RetryPolicy:
    """重试策略：控制失败任务的重试行为。

    max_retries 为最大重试次数（0 表示不重试）。
    backoff_secs 为首次重试延迟（秒）。
    backoff_multiplier 为指数退避倍率（1.0 为固定延迟）。
    """

    max_retries: int = 3
    backoff_secs: float = 1.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise SchedulingError("max_retries must be >= 0")
        if self.backoff_secs < 0:
            raise SchedulingError("backoff_secs must be >= 0")
        if self.backoff_multiplier < 1.0:
            raise SchedulingError("backoff_multiplier must be >= 1.0")

    def delay_for(self, attempt: int) -> float:
        """计算第 attempt 次重试的延迟（秒）。

        attempt 从 1 开始计数。
        """
        if attempt <= 0:
            return 0.0
        return self.backoff_secs * (self.backoff_multiplier ** (attempt - 1))

    def should_retry(self, attempt: int) -> bool:
        """判断当前尝试次数后是否还应重试。

        attempt 为已执行的尝试次数（从 1 开始）。
        """
        return attempt <= self.max_retries


@dataclass(frozen=True)
class SchedulingPolicy:
    """调度策略：约束一次调度轮次的分配行为。

    max_concurrent_per_agent 为单个 Agent 最大并发任务数。
    max_total_assignments 为本轮最大分配总数（None 表示不限）。
    priority_order 为优先级排序方向。
    allow_reassignment 为是否允许任务重新分配给其他 Agent。
    retry_policy 为失败重试策略。
    """

    max_concurrent_per_agent: int = 1
    max_total_assignments: int | None = None
    priority_order: PriorityOrder = PriorityOrder.DESC
    allow_reassignment: bool = False
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if self.max_concurrent_per_agent < 1:
            raise SchedulingError("max_concurrent_per_agent must be >= 1")
        if self.max_total_assignments is not None and self.max_total_assignments < 0:
            raise SchedulingError("max_total_assignments must be >= 0")


@dataclass(frozen=True)
class TaskCandidate:
    """待调度任务候选：携带依赖与重试状态。

    task_id 为 TaskStore 中的任务标识。
    spec 为原始任务规格。
    dependencies_satisfied 为依赖是否全部完成。
    attempt 为已执行尝试次数（从 0 开始，0 表示未执行过）。
    """

    task_id: str
    spec: TaskSpec
    dependencies_satisfied: bool = True
    attempt: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """任务是否可调度：依赖已满足且未被终态标记。"""
        return self.dependencies_satisfied


@dataclass(frozen=True)
class AgentCandidate:
    """可用 Agent 候选：携带负载信息。

    agent_id 为注册标识。
    capabilities 为该 Agent 具备的能力集合。
    current_load 为当前已分配任务数。
    max_concurrent 为该 Agent 允许的最大并发数。
    """

    agent_id: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    current_load: int = 0
    max_concurrent: int = 1

    @property
    def has_capacity(self) -> bool:
        """是否还有并发余量。"""
        return self.current_load < self.max_concurrent

    @property
    def remaining_capacity(self) -> int:
        """剩余并发余量。"""
        return max(0, self.max_concurrent - self.current_load)

    def can_serve(self, required_capabilities: frozenset[str]) -> bool:
        """是否具备指定能力且有并发余量。"""
        return self.has_capacity and required_capabilities.issubset(
            self.capabilities
        )


@dataclass(frozen=True)
class SchedulingResult:
    """一次调度轮次的结果。

    assignments 为成功分配列表。
    deferred 为依赖未满足而推迟的任务 ID 列表。
    rejected 为无可用 Agent 而拒绝的任务 ID 列表。
    retried 为安排重试的任务 ID 列表。
    """

    assignments: tuple[Assignment, ...] = ()
    deferred: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    retried: tuple[str, ...] = ()

    @property
    def total_assigned(self) -> int:
        return len(self.assignments)

    @property
    def total_unassigned(self) -> int:
        return len(self.deferred) + len(self.rejected) + len(self.retried)
