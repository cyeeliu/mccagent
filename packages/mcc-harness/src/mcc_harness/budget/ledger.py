"""AgentBudget 核心预算服务。

提供预留（reserve）、扣减（charge）、结算（settle）、
释放（release）和超限通知（on_exceeded）。

并发安全：所有操作通过 threading.Lock 串行化，
保证多 Agent 共享预算时不会超发。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from mcc_harness.budget.base import (
    BudgetDimension,
    BudgetExceededError,
    BudgetPolicy,
    BudgetRepository,
    BudgetSnapshot,
    UsageEvent,
)


@dataclass
class Reservation:
    """预留句柄：跟踪预分配额度与关联 agent。

    reserve() 成功时返回，release()/settle() 时标记 released。
    """

    agent_id: str
    reserved_tokens: int = 0
    reserved_cost_usd: float = 0.0
    created_at: float = field(default_factory=time.time)
    released: bool = False


@dataclass
class Settlement:
    """结算结果：预留与实际消耗的差额。"""

    agent_id: str
    reserved_tokens: int
    consumed_tokens: int
    reclaimed_tokens: int
    consumed_cost_usd: float
    reclaimed_cost_usd: float


_ExceededCallback = Callable[[BudgetExceededError], None]


class AgentBudget:
    """独立预算服务：多维度限额 + 预留/扣减/结算/释放。

    支持五个维度：tokens、cost_usd、duration_secs、iterations、concurrency。
    所有操作线程安全（threading.Lock）。

    用法::

        budget = AgentBudget("team-1", BudgetPolicy(max_tokens=100_000))
        res = budget.reserve("agent-1", tokens=40_000)
        if res is None:
            ...  # 余额不足
        budget.charge(UsageEvent(agent_id="agent-1", model="m", tokens=15_000))
        settlement = budget.settle("agent-1")

    超限处理::

        budget.on_exceeded(lambda err: log.warning(str(err)))
        budget.charge(event)  # 超限时回调被调用，随后抛出 BudgetExceededError
    """

    def __init__(
        self,
        budget_id: str,
        policy: BudgetPolicy,
        *,
        repository: BudgetRepository | None = None,
    ) -> None:
        if not budget_id:
            raise ValueError("budget_id must not be empty")
        self._budget_id = budget_id
        self._policy = policy
        self._repository = repository

        self._consumed_tokens = 0
        self._consumed_cost_usd = 0.0
        self._consumed_duration_secs = 0.0
        self._consumed_iterations = 0

        self._reservations: dict[str, Reservation] = {}
        self._events: list[UsageEvent] = []
        self._callbacks: list[_ExceededCallback] = []
        self._lock = threading.Lock()

        if repository is not None:
            snapshot = repository.load_snapshot(budget_id)
            if snapshot is not None:
                self._consumed_tokens = snapshot.consumed_tokens
                self._consumed_cost_usd = snapshot.consumed_cost_usd
                self._consumed_duration_secs = snapshot.consumed_duration_secs
                self._consumed_iterations = snapshot.consumed_iterations

    @property
    def budget_id(self) -> str:
        return self._budget_id

    @property
    def policy(self) -> BudgetPolicy:
        return self._policy

    def reserve(
        self,
        agent_id: str,
        *,
        tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> Reservation | None:
        """预留额度。

        检查预留总额 + 本次请求是否超过上限。
        余额不足时返回 None（不抛异常，由调用方决定降级策略）。
        同一 agent_id 重复 reserve 会累加预留量。

        Args:
            agent_id: 关联 Agent 标识
            tokens: 预留 token 数
            cost_usd: 预留美元成本

        Returns:
            预留句柄，或 None 表示余额不足
        """
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if tokens < 0 or cost_usd < 0:
            raise ValueError("reserve amounts must be non-negative")

        with self._lock:
            total_reserved_tokens = sum(
                r.reserved_tokens
                for r in self._reservations.values()
                if not r.released
            )
            total_reserved_cost = sum(
                r.reserved_cost_usd
                for r in self._reservations.values()
                if not r.released
            )
            active = sum(
                1 for r in self._reservations.values() if not r.released
            )

            if (
                self._policy.max_tokens is not None
                and total_reserved_tokens + tokens > self._policy.max_tokens
            ):
                return None
            if (
                self._policy.max_cost_usd is not None
                and total_reserved_cost + cost_usd > self._policy.max_cost_usd
            ):
                return None
            if (
                self._policy.max_concurrency is not None
                and agent_id not in self._reservations
                and active + 1 > self._policy.max_concurrency
            ):
                return None

            existing = self._reservations.get(agent_id)
            if existing is not None and not existing.released:
                existing.reserved_tokens += tokens
                existing.reserved_cost_usd += cost_usd
                return existing

            res = Reservation(
                agent_id=agent_id,
                reserved_tokens=tokens,
                reserved_cost_usd=cost_usd,
            )
            self._reservations[agent_id] = res
            return res

    def charge(self, event: UsageEvent) -> None:
        """扣减实际消耗。

        先检查是否超限，通过后才累加消耗量并记录事件。
        超限时依次调用已注册的 on_exceeded 回调，随后抛出 BudgetExceededError。
        超限的消耗不会被记录，保证 consumed 永不超过 limit。

        Args:
            event: 用量事件

        Raises:
            BudgetExceededError: 任一维度超限
        """
        with self._lock:
            new_tokens = self._consumed_tokens + event.tokens
            new_cost = self._consumed_cost_usd + event.cost_usd
            new_duration = self._consumed_duration_secs + event.duration_secs
            new_iterations = self._consumed_iterations + event.iterations

            errors: list[BudgetExceededError] = []
            if (
                self._policy.max_tokens is not None
                and new_tokens > self._policy.max_tokens
            ):
                errors.append(
                    BudgetExceededError(
                        BudgetDimension.TOKENS,
                        new_tokens,
                        self._policy.max_tokens,
                        event.agent_id,
                    )
                )
            if (
                self._policy.max_cost_usd is not None
                and new_cost > self._policy.max_cost_usd
            ):
                errors.append(
                    BudgetExceededError(
                        BudgetDimension.COST_USD,
                        new_cost,
                        self._policy.max_cost_usd,
                        event.agent_id,
                    )
                )
            if (
                self._policy.max_duration_secs is not None
                and new_duration > self._policy.max_duration_secs
            ):
                errors.append(
                    BudgetExceededError(
                        BudgetDimension.DURATION_SECS,
                        new_duration,
                        self._policy.max_duration_secs,
                        event.agent_id,
                    )
                )
            if (
                self._policy.max_iterations is not None
                and new_iterations > self._policy.max_iterations
            ):
                errors.append(
                    BudgetExceededError(
                        BudgetDimension.ITERATIONS,
                        new_iterations,
                        self._policy.max_iterations,
                        event.agent_id,
                    )
                )

            if errors:
                first_error = errors[0]
            else:
                self._consumed_tokens = new_tokens
                self._consumed_cost_usd = new_cost
                self._consumed_duration_secs = new_duration
                self._consumed_iterations = new_iterations
                self._events.append(event)

                if self._repository is not None:
                    self._repository.append_event(self._budget_id, event)
                first_error = None

        if first_error is not None:
            for cb in self._callbacks:
                cb(first_error)
            raise first_error

        self._persist_snapshot()

    def settle(self, agent_id: str) -> Settlement:
        """结算某 agent 的预留与实际消耗差。

        回收未使用的预留额度（reserved - consumed_for_agent）。
        标记预留为已释放。

        Args:
            agent_id: 关联 Agent 标识

        Returns:
            结算结果
        """
        with self._lock:
            res = self._reservations.get(agent_id)
            if res is None:
                return Settlement(
                    agent_id=agent_id,
                    reserved_tokens=0,
                    consumed_tokens=0,
                    reclaimed_tokens=0,
                    consumed_cost_usd=0.0,
                    reclaimed_cost_usd=0.0,
                )

            agent_events = [e for e in self._events if e.agent_id == agent_id]
            consumed_tokens = sum(e.tokens for e in agent_events)
            consumed_cost = sum(e.cost_usd for e in agent_events)

            reclaimed_tokens = max(0, res.reserved_tokens - consumed_tokens)
            reclaimed_cost = max(0.0, res.reserved_cost_usd - consumed_cost)

            res.released = True

            result = Settlement(
                agent_id=agent_id,
                reserved_tokens=res.reserved_tokens,
                consumed_tokens=consumed_tokens,
                reclaimed_tokens=reclaimed_tokens,
                consumed_cost_usd=consumed_cost,
                reclaimed_cost_usd=reclaimed_cost,
            )

        self._persist_snapshot()
        return result

    def release(self, agent_id: str) -> Reservation | None:
        """释放预留额度，不结算消耗。

        用于 agent 异常退出时回收预留。
        已消耗量不变，仅标记预留为已释放。

        Args:
            agent_id: 关联 Agent 标识

        Returns:
            被释放的预留句柄，或 None 表示无预留
        """
        with self._lock:
            res = self._reservations.get(agent_id)
            if res is not None:
                res.released = True
            return res

    def snapshot(self) -> BudgetSnapshot:
        """获取当前预算快照（不可变视图）。"""
        with self._lock:
            active = sum(
                1 for r in self._reservations.values() if not r.released
            )
            reserved_tokens = sum(
                r.reserved_tokens
                for r in self._reservations.values()
                if not r.released
            )
            reserved_cost = sum(
                r.reserved_cost_usd
                for r in self._reservations.values()
                if not r.released
            )
            return BudgetSnapshot(
                policy=self._policy,
                consumed_tokens=self._consumed_tokens,
                consumed_cost_usd=self._consumed_cost_usd,
                consumed_duration_secs=self._consumed_duration_secs,
                consumed_iterations=self._consumed_iterations,
                active_concurrency=active,
                reserved_tokens=reserved_tokens,
                reserved_cost_usd=reserved_cost,
            )

    def on_exceeded(self, callback: _ExceededCallback) -> None:
        """注册超限回调。

        回调在 charge() 检测到超限时、抛出异常前被调用。
        可注册多个回调，按注册顺序依次调用。

        Args:
            callback: 接收 BudgetExceededError 的回调函数
        """
        self._callbacks.append(callback)

    def events(self) -> list[UsageEvent]:
        """返回全部用量事件副本。"""
        with self._lock:
            return list(self._events)

    def is_exceeded(self) -> bool:
        """是否已超限。"""
        return self.snapshot().is_exceeded

    def remaining_tokens(self) -> int | None:
        """剩余 token 额度，None 表示不限。"""
        return self.snapshot().remaining_tokens

    def remaining_cost_usd(self) -> float | None:
        """剩余美元额度，None 表示不限。"""
        return self.snapshot().remaining_cost_usd

    def summary_by_agent(self) -> dict[str, UsageEvent]:
        """按 agent_id 汇总用量。"""
        with self._lock:
            result: dict[str, UsageEvent] = {}
            for e in self._events:
                if e.agent_id not in result:
                    result[e.agent_id] = UsageEvent(
                        agent_id=e.agent_id,
                        model=e.model,
                        task_id=e.task_id,
                        tokens=e.tokens,
                        cost_usd=e.cost_usd,
                        duration_secs=e.duration_secs,
                        iterations=e.iterations,
                        timestamp=e.timestamp,
                    )
                else:
                    prev = result[e.agent_id]
                    result[e.agent_id] = UsageEvent(
                        agent_id=e.agent_id,
                        model=e.model,
                        task_id=e.task_id,
                        tokens=prev.tokens + e.tokens,
                        cost_usd=prev.cost_usd + e.cost_usd,
                        duration_secs=prev.duration_secs + e.duration_secs,
                        iterations=prev.iterations + e.iterations,
                        timestamp=e.timestamp,
                    )
            return result

    def summary_by_model(self) -> dict[str, UsageEvent]:
        """按 model 汇总用量。"""
        with self._lock:
            result: dict[str, UsageEvent] = {}
            for e in self._events:
                if e.model not in result:
                    result[e.model] = UsageEvent(
                        agent_id="*",
                        model=e.model,
                        tokens=e.tokens,
                        cost_usd=e.cost_usd,
                        duration_secs=e.duration_secs,
                        iterations=e.iterations,
                        timestamp=e.timestamp,
                    )
                else:
                    prev = result[e.model]
                    result[e.model] = UsageEvent(
                        agent_id="*",
                        model=e.model,
                        tokens=prev.tokens + e.tokens,
                        cost_usd=prev.cost_usd + e.cost_usd,
                        duration_secs=prev.duration_secs + e.duration_secs,
                        iterations=prev.iterations + e.iterations,
                        timestamp=e.timestamp,
                    )
            return result

    def count(self) -> int:
        """已记录的用量事件数。"""
        with self._lock:
            return len(self._events)

    def clear(self) -> None:
        """清空所有状态（预留、消耗、事件）。

        主要用于测试重置。
        """
        with self._lock:
            self._reservations.clear()
            self._events.clear()
            self._consumed_tokens = 0
            self._consumed_cost_usd = 0.0
            self._consumed_duration_secs = 0.0
            self._consumed_iterations = 0

    def _persist_snapshot(self) -> None:
        """持久化当前快照到 repository。"""
        if self._repository is not None:
            self._repository.save_snapshot(self._budget_id, self.snapshot())
