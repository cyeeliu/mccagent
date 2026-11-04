"""AgentBudget 独立预算服务测试。

覆盖：
- BudgetPolicy 校验与维度查询
- UsageEvent 校验
- BudgetSnapshot 剩余/超限计算与序列化
- AgentBudget 预留/扣减/结算/释放/快照
- 并发安全（多线程共享预算不超发）
- 超限回调通知
- 持久化（InMemoryBudgetRepository）
- 汇总（by agent / by model）
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mcc_harness.budget import (
    AgentBudget,
    BudgetDimension,
    BudgetExceededError,
    BudgetPolicy,
    BudgetPolicyError,
    BudgetSnapshot,
    InMemoryBudgetRepository,
    Reservation,
    Settlement,
    UsageEvent,
)


class TestBudgetPolicy:
    """BudgetPolicy 校验与属性测试。"""

    def test_defaults_all_none(self):
        p = BudgetPolicy()
        assert p.max_tokens is None
        assert p.is_unlimited is True
        assert p.dimensions == frozenset()

    def test_configured_dimensions(self):
        p = BudgetPolicy(max_tokens=1000, max_cost_usd=5.0)
        assert p.is_unlimited is False
        assert BudgetDimension.TOKENS in p.dimensions
        assert BudgetDimension.COST_USD in p.dimensions
        assert BudgetDimension.DURATION_SECS not in p.dimensions

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_tokens": -1},
            {"max_cost_usd": -0.1},
            {"max_duration_secs": -1.0},
            {"max_iterations": -1},
            {"max_concurrency": -1},
        ],
    )
    def test_negative_limit_rejected(self, kwargs):
        with pytest.raises(BudgetPolicyError):
            BudgetPolicy(**kwargs)

    def test_zero_limits_allowed(self):
        p = BudgetPolicy(max_tokens=0, max_concurrency=0)
        assert p.max_tokens == 0
        assert p.max_concurrency == 0

    def test_frozen(self):
        p = BudgetPolicy(max_tokens=100)
        with pytest.raises(AttributeError):
            p.max_tokens = 200  # type: ignore[misc]


class TestUsageEvent:
    """UsageEvent 校验测试。"""

    def test_minimal(self):
        e = UsageEvent(agent_id="a", model="m")
        assert e.tokens == 0
        assert e.cost_usd == 0.0
        assert e.timestamp > 0

    def test_full(self):
        e = UsageEvent(
            agent_id="a", model="m", task_id="t1",
            tokens=100, cost_usd=0.01, duration_secs=1.5, iterations=3,
        )
        assert e.task_id == "t1"
        assert e.tokens == 100
        assert e.iterations == 3

    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            ({"agent_id": "", "model": "m"}, "agent_id"),
            ({"agent_id": "a", "model": ""}, "model"),
            ({"agent_id": "a", "model": "m", "tokens": -1}, "tokens"),
            ({"agent_id": "a", "model": "m", "cost_usd": -0.1}, "cost_usd"),
            ({"agent_id": "a", "model": "m", "duration_secs": -1}, "duration"),
            ({"agent_id": "a", "model": "m", "iterations": -1}, "iterations"),
        ],
    )
    def test_invalid_rejected(self, kwargs, msg):
        with pytest.raises(BudgetPolicyError, match=msg):
            UsageEvent(**kwargs)


class TestBudgetSnapshot:
    """BudgetSnapshot 剩余/超限/序列化测试。"""

    def test_remaining_with_limits(self):
        s = BudgetSnapshot(
            policy=BudgetPolicy(max_tokens=1000, max_cost_usd=10.0),
            consumed_tokens=300,
            consumed_cost_usd=3.0,
        )
        assert s.remaining_tokens == 700
        assert s.remaining_cost_usd == pytest.approx(7.0)

    def test_remaining_unlimited(self):
        s = BudgetSnapshot(policy=BudgetPolicy())
        assert s.remaining_tokens is None
        assert s.remaining_cost_usd is None

    def test_remaining_floored_at_zero(self):
        s = BudgetSnapshot(
            policy=BudgetPolicy(max_tokens=100),
            consumed_tokens=150,
        )
        assert s.remaining_tokens == 0

    @pytest.mark.parametrize(
        "consumed,limit,expected",
        [
            (50, 100, False),
            (100, 100, False),
            (101, 100, True),
        ],
    )
    def test_is_exceeded_tokens(self, consumed, limit, expected):
        s = BudgetSnapshot(
            policy=BudgetPolicy(max_tokens=limit),
            consumed_tokens=consumed,
        )
        assert s.is_exceeded is expected

    def test_is_exceeded_concurrency(self):
        s = BudgetSnapshot(
            policy=BudgetPolicy(max_concurrency=2),
            active_concurrency=3,
        )
        assert s.is_exceeded is True

    def test_to_dict(self):
        s = BudgetSnapshot(
            policy=BudgetPolicy(max_tokens=100),
            consumed_tokens=30,
            active_concurrency=2,
        )
        d = s.to_dict()
        assert d["consumed_tokens"] == 30
        assert d["active_concurrency"] == 2
        assert d["policy"]["max_tokens"] == 100


class TestAgentBudgetReserve:
    """AgentBudget.reserve 测试。"""

    def test_reserve_success(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        res = b.reserve("agent-1", tokens=40_000)
        assert res is not None
        assert res.agent_id == "agent-1"
        assert res.reserved_tokens == 40_000
        assert res.released is False

    def test_reserve_insufficient(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=50_000))
        assert b.reserve("a1", tokens=30_000) is not None
        assert b.reserve("a2", tokens=30_000) is None

    def test_reserve_exact_fit(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=60_000))
        assert b.reserve("a1", tokens=30_000) is not None
        assert b.reserve("a2", tokens=30_000) is not None

    def test_reserve_accumulate_same_agent(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        r1 = b.reserve("a1", tokens=30_000)
        r2 = b.reserve("a1", tokens=20_000)
        assert r1 is not None and r2 is not None
        assert r2.reserved_tokens == 50_000

    def test_reserve_cost_limit(self):
        b = AgentBudget("b1", BudgetPolicy(max_cost_usd=10.0))
        assert b.reserve("a1", cost_usd=6.0) is not None
        assert b.reserve("a2", cost_usd=6.0) is None

    def test_reserve_concurrency_limit(self):
        b = AgentBudget("b1", BudgetPolicy(max_concurrency=2))
        assert b.reserve("a1") is not None
        assert b.reserve("a2") is not None
        assert b.reserve("a3") is None

    def test_reserve_concurrency_same_agent_ok(self):
        b = AgentBudget("b1", BudgetPolicy(max_concurrency=1))
        assert b.reserve("a1") is not None
        assert b.reserve("a1") is not None

    def test_reserve_after_release(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=60_000)
        b.release("a1")
        res = b.reserve("a2", tokens=60_000)
        assert res is not None

    def test_reserve_empty_agent_rejected(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        with pytest.raises(ValueError, match="agent_id"):
            b.reserve("", tokens=10)

    def test_reserve_negative_rejected(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        with pytest.raises(ValueError, match="non-negative"):
            b.reserve("a1", tokens=-1)


class TestAgentBudgetCharge:
    """AgentBudget.charge 测试。"""

    def test_charge_success(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=1000))
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=300))
        snap = b.snapshot()
        assert snap.consumed_tokens == 300

    def test_charge_accumulate(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=1000))
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=100))
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=200))
        assert b.snapshot().consumed_tokens == 300

    def test_charge_exceeds_tokens(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        with pytest.raises(BudgetExceededError) as exc_info:
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=150))
        assert exc_info.value.dimension == BudgetDimension.TOKENS
        assert exc_info.value.agent_id == "a1"

    def test_charge_exceeds_cost(self):
        b = AgentBudget("b1", BudgetPolicy(max_cost_usd=1.0))
        with pytest.raises(BudgetExceededError) as exc_info:
            b.charge(UsageEvent(agent_id="a1", model="m", cost_usd=1.5))
        assert exc_info.value.dimension == BudgetDimension.COST_USD

    def test_charge_exceeds_duration(self):
        b = AgentBudget("b1", BudgetPolicy(max_duration_secs=10.0))
        with pytest.raises(BudgetExceededError) as exc_info:
            b.charge(
                UsageEvent(agent_id="a1", model="m", duration_secs=15.0)
            )
        assert exc_info.value.dimension == BudgetDimension.DURATION_SECS

    def test_charge_exceeds_iterations(self):
        b = AgentBudget("b1", BudgetPolicy(max_iterations=5))
        with pytest.raises(BudgetExceededError) as exc_info:
            b.charge(
                UsageEvent(agent_id="a1", model="m", iterations=6)
            )
        assert exc_info.value.dimension == BudgetDimension.ITERATIONS

    def test_charge_exact_limit_ok(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=100))
        assert b.snapshot().consumed_tokens == 100

    def test_charge_multiple_dimensions(self):
        b = AgentBudget(
            "b1",
            BudgetPolicy(max_tokens=100, max_cost_usd=1.0),
        )
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=50, cost_usd=0.5))
        with pytest.raises(BudgetExceededError):
            b.charge(
                UsageEvent(agent_id="a1", model="m", tokens=60, cost_usd=0.6)
            )

    def test_charge_not_recorded_on_exceed(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        with pytest.raises(BudgetExceededError):
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=150))
        assert b.count() == 0
        assert b.snapshot().consumed_tokens == 0


class TestAgentBudgetExceededCallback:
    """超限回调通知测试。"""

    def test_callback_called_before_raise(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        received: list[BudgetExceededError] = []
        b.on_exceeded(received.append)
        with pytest.raises(BudgetExceededError):
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=150))
        assert len(received) == 1
        assert received[0].dimension == BudgetDimension.TOKENS

    def test_multiple_callbacks_in_order(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        order: list[str] = []
        b.on_exceeded(lambda e: order.append("first"))
        b.on_exceeded(lambda e: order.append("second"))
        with pytest.raises(BudgetExceededError):
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=150))
        assert order == ["first", "second"]

    def test_no_callback_still_raises(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        with pytest.raises(BudgetExceededError):
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=150))


class TestAgentBudgetSettle:
    """AgentBudget.settle 结算测试。"""

    def test_settle_reclaims_unused(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=50_000)
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=20_000))
        s = b.settle("a1")
        assert isinstance(s, Settlement)
        assert s.reserved_tokens == 50_000
        assert s.consumed_tokens == 20_000
        assert s.reclaimed_tokens == 30_000

    def test_settle_no_reservation(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        s = b.settle("nonexistent")
        assert s.reclaimed_tokens == 0
        assert s.consumed_tokens == 0

    def test_settle_marks_released(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=50_000)
        b.settle("a1")
        snap = b.snapshot()
        assert snap.reserved_tokens == 0

    def test_settle_cost_reclaim(self):
        b = AgentBudget("b1", BudgetPolicy(max_cost_usd=100.0))
        b.reserve("a1", cost_usd=50.0)
        b.charge(UsageEvent(agent_id="a1", model="m", cost_usd=15.0))
        s = b.settle("a1")
        assert s.reclaimed_cost_usd == pytest.approx(35.0)


class TestAgentBudgetRelease:
    """AgentBudget.release 释放测试。"""

    def test_release_returns_reservation(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=40_000)
        res = b.release("a1")
        assert res is not None
        assert res.released is True

    def test_release_nonexistent(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        assert b.release("nope") is None

    def test_release_frees_capacity(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=60_000)
        b.release("a1")
        assert b.reserve("a2", tokens=60_000) is not None

    def test_release_preserves_consumed(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100_000))
        b.reserve("a1", tokens=50_000)
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=20_000))
        b.release("a1")
        assert b.snapshot().consumed_tokens == 20_000


class TestAgentBudgetSnapshot:
    """AgentBudget.snapshot 快照测试。"""

    def test_snapshot_initial(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=1000))
        s = b.snapshot()
        assert s.consumed_tokens == 0
        assert s.active_concurrency == 0
        assert s.reserved_tokens == 0

    def test_snapshot_after_reserve_and_charge(self):
        b = AgentBudget(
            "b1",
            BudgetPolicy(max_tokens=1000, max_concurrency=3),
        )
        b.reserve("a1", tokens=300)
        b.reserve("a2", tokens=200)
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=100))
        s = b.snapshot()
        assert s.reserved_tokens == 500
        assert s.active_concurrency == 2
        assert s.consumed_tokens == 100

    def test_is_exceeded(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=100))
        assert b.is_exceeded() is False
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=100))
        assert b.is_exceeded() is False
        with pytest.raises(BudgetExceededError):
            b.charge(UsageEvent(agent_id="a1", model="m", tokens=1))
        assert b.is_exceeded() is False

    def test_remaining_tokens(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=1000))
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=300))
        assert b.remaining_tokens() == 700

    def test_remaining_tokens_unlimited(self):
        b = AgentBudget("b1", BudgetPolicy())
        assert b.remaining_tokens() is None


class TestAgentBudgetSummary:
    """AgentBudget 汇总测试。"""

    def test_summary_by_agent(self):
        b = AgentBudget("b1", BudgetPolicy())
        b.charge(UsageEvent(agent_id="a1", model="m1", tokens=100, cost_usd=0.1))
        b.charge(UsageEvent(agent_id="a1", model="m1", tokens=200, cost_usd=0.2))
        b.charge(UsageEvent(agent_id="a2", model="m2", tokens=50, cost_usd=0.05))
        summary = b.summary_by_agent()
        assert summary["a1"].tokens == 300
        assert summary["a1"].cost_usd == pytest.approx(0.3)
        assert summary["a2"].tokens == 50

    def test_summary_by_model(self):
        b = AgentBudget("b1", BudgetPolicy())
        b.charge(UsageEvent(agent_id="a1", model="m1", tokens=100))
        b.charge(UsageEvent(agent_id="a2", model="m1", tokens=200))
        b.charge(UsageEvent(agent_id="a1", model="m2", tokens=50))
        summary = b.summary_by_model()
        assert summary["m1"].tokens == 300
        assert summary["m2"].tokens == 50

    def test_events_list(self):
        b = AgentBudget("b1", BudgetPolicy())
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=10))
        b.charge(UsageEvent(agent_id="a2", model="m", tokens=20))
        events = b.events()
        assert len(events) == 2
        assert events[0].agent_id == "a1"

    def test_count(self):
        b = AgentBudget("b1", BudgetPolicy())
        assert b.count() == 0
        b.charge(UsageEvent(agent_id="a1", model="m"))
        assert b.count() == 1


class TestAgentBudgetConcurrency:
    """并发安全测试：多线程共享预算不超发。"""

    def test_concurrent_charge_no_overspend(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=10_000))
        num_threads = 20
        per_thread = 100

        def worker(tid: int) -> int:
            charged = 0
            for _ in range(per_thread):
                try:
                    b.charge(
                        UsageEvent(
                            agent_id=f"a{tid}", model="m", tokens=10
                        )
                    )
                    charged += 1
                except BudgetExceededError:
                    break
            return charged

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = list(pool.map(worker, range(num_threads)))

        total_charged = sum(results)
        assert total_charged <= 1000
        assert b.snapshot().consumed_tokens == total_charged * 10
        assert b.snapshot().consumed_tokens <= 10_000

    def test_concurrent_reserve_no_oversell(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=10_000))
        num_threads = 50
        success_count = [0]
        lock = threading.Lock()

        def worker(tid: int) -> None:
            res = b.reserve(f"a{tid}", tokens=500)
            if res is not None:
                with lock:
                    success_count[0] += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count[0] <= 20
        assert b.snapshot().reserved_tokens <= 10_000

    def test_concurrent_mixed_operations(self):
        b = AgentBudget(
            "b1",
            BudgetPolicy(max_tokens=50_000, max_concurrency=10),
        )
        errors: list[Exception] = []

        def worker(tid: int) -> None:
            try:
                res = b.reserve(f"a{tid}", tokens=1000)
                if res is None:
                    return
                b.charge(
                    UsageEvent(agent_id=f"a{tid}", model="m", tokens=500)
                )
                b.settle(f"a{tid}")
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=30) as pool:
            list(pool.map(worker, range(30)))

        assert len(errors) == 0
        assert b.snapshot().consumed_tokens <= 50_000


class TestAgentBudgetPersistence:
    """持久化测试：InMemoryBudgetRepository。"""

    def test_repository_save_load_snapshot(self):
        repo = InMemoryBudgetRepository()
        snap = BudgetSnapshot(
            policy=BudgetPolicy(max_tokens=1000),
            consumed_tokens=300,
        )
        repo.save_snapshot("b1", snap)
        loaded = repo.load_snapshot("b1")
        assert loaded is not None
        assert loaded.consumed_tokens == 300

    def test_repository_load_nonexistent(self):
        repo = InMemoryBudgetRepository()
        assert repo.load_snapshot("nope") is None

    def test_repository_append_list_events(self):
        repo = InMemoryBudgetRepository()
        e1 = UsageEvent(agent_id="a1", model="m", tokens=10)
        e2 = UsageEvent(agent_id="a2", model="m", tokens=20)
        repo.append_event("b1", e1)
        repo.append_event("b1", e2)
        events = repo.list_events("b1")
        assert len(events) == 2
        assert events[0].tokens == 10

    def test_budget_with_repository_persists_events(self):
        repo = InMemoryBudgetRepository()
        b = AgentBudget("b1", BudgetPolicy(max_tokens=10_000), repository=repo)
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=100))
        events = repo.list_events("b1")
        assert len(events) == 1
        assert events[0].tokens == 100

    def test_budget_restores_from_repository(self):
        repo = InMemoryBudgetRepository()
        b1 = AgentBudget("b1", BudgetPolicy(max_tokens=10_000), repository=repo)
        b1.charge(UsageEvent(agent_id="a1", model="m", tokens=300))
        b1.charge(UsageEvent(agent_id="a1", model="m", cost_usd=0.5))

        b2 = AgentBudget("b1", BudgetPolicy(max_tokens=10_000), repository=repo)
        snap = b2.snapshot()
        assert snap.consumed_tokens == 300
        assert snap.consumed_cost_usd == pytest.approx(0.5)


class TestAgentBudgetMisc:
    """其他边界测试。"""

    def test_empty_budget_id_rejected(self):
        with pytest.raises(ValueError, match="budget_id"):
            AgentBudget("", BudgetPolicy())

    def test_clear(self):
        b = AgentBudget("b1", BudgetPolicy(max_tokens=1000))
        b.reserve("a1", tokens=100)
        b.charge(UsageEvent(agent_id="a1", model="m", tokens=50))
        b.clear()
        assert b.count() == 0
        assert b.snapshot().consumed_tokens == 0
        assert b.snapshot().reserved_tokens == 0

    def test_budget_id_property(self):
        b = AgentBudget("team-1", BudgetPolicy())
        assert b.budget_id == "team-1"

    def test_policy_property(self):
        p = BudgetPolicy(max_tokens=100)
        b = AgentBudget("b1", p)
        assert b.policy is p

    def test_reservation_is_dataclass(self):
        res = Reservation(agent_id="a1", reserved_tokens=100)
        assert res.agent_id == "a1"
        assert res.reserved_tokens == 100
        assert res.released is False

    def test_settlement_is_dataclass(self):
        s = Settlement(
            agent_id="a1",
            reserved_tokens=100,
            consumed_tokens=60,
            reclaimed_tokens=40,
            consumed_cost_usd=0.3,
            reclaimed_cost_usd=0.2,
        )
        assert s.reclaimed_tokens == 40
