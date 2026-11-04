"""AgentBudget 独立预算服务：类型定义与持久化协议。

将预算核算从 SubagentRunner 中抽离，作为 Runtime 可复用服务。
支持 Token、成本、时长、迭代数和并发数五维限制。

预算策略与模型价格表解耦：BudgetPolicy 只声明各维度上限，
价格换算由调用方在构造 UsageEvent 时填入 cost_usd。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class BudgetDimension(StrEnum):
    """预算维度枚举。"""

    TOKENS = "tokens"
    COST_USD = "cost_usd"
    DURATION_SECS = "duration_secs"
    ITERATIONS = "iterations"
    CONCURRENCY = "concurrency"


class BudgetExceededError(ValueError):
    """预算超限错误。

    携带超限维度、已消耗量、上限值和关联 agent_id，
    便于上层做精细化处理（如降级模型、终止特定 agent）。
    """

    def __init__(
        self,
        dimension: BudgetDimension,
        consumed: float,
        limit: float,
        agent_id: str | None = None,
    ) -> None:
        self.dimension = dimension
        self.consumed = consumed
        self.limit = limit
        self.agent_id = agent_id
        agent_part = f" (agent={agent_id})" if agent_id else ""
        super().__init__(
            f"budget exceeded{agent_part}: "
            f"{dimension.value} {consumed} > {limit}"
        )


class BudgetPolicyError(ValueError):
    """预算策略配置错误。"""


@dataclass(frozen=True)
class BudgetPolicy:
    """预算策略：各维度的上限。

    None 表示该维度不限。所有维度均为 None 时为无限预算。
    与模型价格表解耦——价格换算由调用方在 UsageEvent.cost_usd 填入。
    """

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_duration_secs: float | None = None
    max_iterations: int | None = None
    max_concurrency: int | None = None

    def __post_init__(self) -> None:
        for dim, val in [
            (BudgetDimension.TOKENS, self.max_tokens),
            (BudgetDimension.ITERATIONS, self.max_iterations),
            (BudgetDimension.CONCURRENCY, self.max_concurrency),
        ]:
            if val is not None and val < 0:
                raise BudgetPolicyError(
                    f"{dim.value} limit must be non-negative, got {val}"
                )
        for dim, val in [
            (BudgetDimension.COST_USD, self.max_cost_usd),
            (BudgetDimension.DURATION_SECS, self.max_duration_secs),
        ]:
            if val is not None and val < 0:
                raise BudgetPolicyError(
                    f"{dim.value} limit must be non-negative, got {val}"
                )

    @property
    def is_unlimited(self) -> bool:
        """所有维度均不限。"""
        return all(
            v is None
            for v in (
                self.max_tokens,
                self.max_cost_usd,
                self.max_duration_secs,
                self.max_iterations,
                self.max_concurrency,
            )
        )

    @property
    def dimensions(self) -> frozenset[BudgetDimension]:
        """已配置上限的维度集合。"""
        mapping: list[tuple[BudgetDimension, int | float | None]] = [
            (BudgetDimension.TOKENS, self.max_tokens),
            (BudgetDimension.COST_USD, self.max_cost_usd),
            (BudgetDimension.DURATION_SECS, self.max_duration_secs),
            (BudgetDimension.ITERATIONS, self.max_iterations),
            (BudgetDimension.CONCURRENCY, self.max_concurrency),
        ]
        return frozenset(dim for dim, val in mapping if val is not None)


@dataclass
class UsageEvent:
    """用量事件：单次资源消耗记录。

    每条记录关联 agent_id/task_id/model，满足可观测性要求。
    非负约束在创建时校验。
    """

    agent_id: str
    model: str
    task_id: str | None = None
    tokens: int = 0
    cost_usd: float = 0.0
    duration_secs: float = 0.0
    iterations: int = 0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise BudgetPolicyError("agent_id must not be empty")
        if not self.model:
            raise BudgetPolicyError("model must not be empty")
        if self.tokens < 0:
            raise BudgetPolicyError(f"tokens must be non-negative, got {self.tokens}")
        if self.cost_usd < 0:
            raise BudgetPolicyError(
                f"cost_usd must be non-negative, got {self.cost_usd}"
            )
        if self.duration_secs < 0:
            raise BudgetPolicyError(
                f"duration_secs must be non-negative, got {self.duration_secs}"
            )
        if self.iterations < 0:
            raise BudgetPolicyError(
                f"iterations must be non-negative, got {self.iterations}"
            )


@dataclass(frozen=True)
class BudgetSnapshot:
    """预算快照：某时刻的预算状态不可变视图。

    用于日志、监控和持久化恢复。
    """

    policy: BudgetPolicy
    consumed_tokens: int = 0
    consumed_cost_usd: float = 0.0
    consumed_duration_secs: float = 0.0
    consumed_iterations: int = 0
    active_concurrency: int = 0
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0

    @property
    def remaining_tokens(self) -> int | None:
        if self.policy.max_tokens is None:
            return None
        return max(0, self.policy.max_tokens - self.consumed_tokens)

    @property
    def remaining_cost_usd(self) -> float | None:
        if self.policy.max_cost_usd is None:
            return None
        return max(0.0, self.policy.max_cost_usd - self.consumed_cost_usd)

    @property
    def remaining_duration_secs(self) -> float | None:
        if self.policy.max_duration_secs is None:
            return None
        return max(0.0, self.policy.max_duration_secs - self.consumed_duration_secs)

    @property
    def remaining_iterations(self) -> int | None:
        if self.policy.max_iterations is None:
            return None
        return max(0, self.policy.max_iterations - self.consumed_iterations)

    @property
    def remaining_concurrency(self) -> int | None:
        if self.policy.max_concurrency is None:
            return None
        return max(0, self.policy.max_concurrency - self.active_concurrency)

    @property
    def is_exceeded(self) -> bool:
        """任一维度超限即为超限。"""
        checks: list[tuple[int | float | None, int | float]] = [
            (self.policy.max_tokens, self.consumed_tokens),
            (self.policy.max_cost_usd, self.consumed_cost_usd),
            (self.policy.max_duration_secs, self.consumed_duration_secs),
            (self.policy.max_iterations, self.consumed_iterations),
            (self.policy.max_concurrency, self.active_concurrency),
        ]
        return any(
            limit is not None and consumed > limit for limit, consumed in checks
        )

    def to_dict(self) -> dict[str, int | float | None | str | list[str]]:
        """序列化为字典，用于日志和恢复。"""
        return {
            "policy": {
                "max_tokens": self.policy.max_tokens,
                "max_cost_usd": self.policy.max_cost_usd,
                "max_duration_secs": self.policy.max_duration_secs,
                "max_iterations": self.policy.max_iterations,
                "max_concurrency": self.policy.max_concurrency,
            },
            "consumed_tokens": self.consumed_tokens,
            "consumed_cost_usd": self.consumed_cost_usd,
            "consumed_duration_secs": self.consumed_duration_secs,
            "consumed_iterations": self.consumed_iterations,
            "active_concurrency": self.active_concurrency,
            "reserved_tokens": self.reserved_tokens,
            "reserved_cost_usd": self.reserved_cost_usd,
        }


class BudgetRepository(Protocol):
    """预算持久化接口。

    支持快照保存/加载和用量事件追加/查询，
    用于跨进程恢复或分布式账本扩展。
    """

    def save_snapshot(self, budget_id: str, snapshot: BudgetSnapshot) -> None:
        """保存预算快照。"""
        ...

    def load_snapshot(self, budget_id: str) -> BudgetSnapshot | None:
        """加载预算快照，不存在返回 None。"""
        ...

    def append_event(self, budget_id: str, event: UsageEvent) -> None:
        """追加用量事件。"""
        ...

    def list_events(self, budget_id: str) -> list[UsageEvent]:
        """查询全部用量事件。"""
        ...


class InMemoryBudgetRepository:
    """内存持久化实现：线程安全，用于单进程场景。

    所有操作通过 threading.Lock 串行化，
    保证多线程并发访问的一致性。
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, BudgetSnapshot] = {}
        self._events: dict[str, list[UsageEvent]] = {}
        self._lock = threading.Lock()

    def save_snapshot(self, budget_id: str, snapshot: BudgetSnapshot) -> None:
        with self._lock:
            self._snapshots[budget_id] = snapshot

    def load_snapshot(self, budget_id: str) -> BudgetSnapshot | None:
        with self._lock:
            return self._snapshots.get(budget_id)

    def append_event(self, budget_id: str, event: UsageEvent) -> None:
        with self._lock:
            self._events.setdefault(budget_id, []).append(event)

    def list_events(self, budget_id: str) -> list[UsageEvent]:
        with self._lock:
            return list(self._events.get(budget_id, []))
