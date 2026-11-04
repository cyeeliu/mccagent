"""TaskScheduler 任务调度器测试：依赖 DAG、能力匹配、并发限制、重试策略。

覆盖 P2-07 验收标准：
- 依赖任务完成后下游任务才可调度
- 同一任务不能重复分配
- 失败任务按策略重试或转人工
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_harness import (
    AgentDescriptor,
    AgentRegistry,
    Assignment,
    PriorityOrder,
    RegistrationStatus,
    RetryHandler,
    SchedulingError,
    SchedulingPolicy,
    SchedulingResult,
    TaskCandidate,
    TaskSchedulerImpl,
    TaskSpec,
)
from mcc_harness.scheduler import LeaseManager, RetryPolicy
from mcc_harness.scheduler.base import RetryExhaustedError
from mcc_tools import TaskStore

# -- fixture --


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry(heartbeat_timeout=30.0)


@pytest.fixture
def task_store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / ".tasks")


def register_agent(
    registry: AgentRegistry,
    agent_id: str,
    capabilities: frozenset[str] = frozenset(),
) -> None:
    descriptor = AgentDescriptor(
        agent_id=agent_id,
        agent_name=agent_id,
        capabilities=capabilities,
    )
    registry.register(descriptor)


def make_spec(
    subject: str = "task",
    capabilities: frozenset[str] = frozenset(),
    priority: int = 0,
    blocked_by: tuple[str, ...] = (),
) -> TaskSpec:
    return TaskSpec(
        subject=subject,
        description=f"desc: {subject}",
        required_capabilities=capabilities,
        priority=priority,
        blocked_by=blocked_by,
    )


# -- SchedulingPolicy 校验 --


class TestSchedulingPolicy:
    """SchedulingPolicy 配置校验。"""

    def test_default_policy(self):
        policy = SchedulingPolicy()
        assert policy.max_concurrent_per_agent == 1
        assert policy.max_total_assignments is None
        assert policy.priority_order == PriorityOrder.DESC
        assert policy.allow_reassignment is False

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_concurrent_per_agent", 0),
            ("max_concurrent_per_agent", -1),
        ],
    )
    def test_invalid_max_concurrent_raises(self, field, value):
        with pytest.raises(SchedulingError):
            SchedulingPolicy(**{field: value})

    def test_invalid_max_total_raises(self):
        with pytest.raises(SchedulingError):
            SchedulingPolicy(max_total_assignments=-1)

    def test_custom_policy(self):
        policy = SchedulingPolicy(
            max_concurrent_per_agent=3,
            max_total_assignments=10,
            priority_order=PriorityOrder.ASC,
            allow_reassignment=True,
        )
        assert policy.max_concurrent_per_agent == 3
        assert policy.max_total_assignments == 10
        assert policy.priority_order == PriorityOrder.ASC
        assert policy.allow_reassignment is True


# -- RetryPolicy --


class TestRetryPolicy:
    """RetryPolicy 重试策略。"""

    def test_default_policy(self):
        rp = RetryPolicy()
        assert rp.max_retries == 3
        assert rp.backoff_secs == 1.0
        assert rp.backoff_multiplier == 2.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_retries", -1),
            ("backoff_secs", -0.1),
            ("backoff_multiplier", 0.5),
        ],
    )
    def test_invalid_raises(self, field, value):
        with pytest.raises(SchedulingError):
            RetryPolicy(**{field: value})

    def test_delay_for_fixed(self):
        rp = RetryPolicy(backoff_secs=2.0, backoff_multiplier=1.0)
        assert rp.delay_for(1) == 2.0
        assert rp.delay_for(2) == 2.0
        assert rp.delay_for(3) == 2.0

    def test_delay_for_exponential(self):
        rp = RetryPolicy(backoff_secs=1.0, backoff_multiplier=2.0)
        assert rp.delay_for(1) == 1.0
        assert rp.delay_for(2) == 2.0
        assert rp.delay_for(3) == 4.0
        assert rp.delay_for(4) == 8.0

    def test_delay_for_zero_attempt(self):
        rp = RetryPolicy()
        assert rp.delay_for(0) == 0.0

    @pytest.mark.parametrize(
        "max_retries,attempt,expected",
        [
            (3, 1, True),
            (3, 2, True),
            (3, 3, True),
            (3, 4, False),
            (0, 1, False),
        ],
    )
    def test_should_retry(self, max_retries, attempt, expected):
        rp = RetryPolicy(max_retries=max_retries)
        assert rp.should_retry(attempt) is expected


# -- TaskCandidate --


class TestTaskCandidate:
    """TaskCandidate 候选任务。"""

    def test_is_ready_default(self):
        tc = TaskCandidate(task_id="t1", spec=make_spec())
        assert tc.is_ready is True

    def test_not_ready_when_deps_unsatisfied(self):
        tc = TaskCandidate(
            task_id="t1",
            spec=make_spec(),
            dependencies_satisfied=False,
        )
        assert tc.is_ready is False

    def test_frozen(self):
        tc = TaskCandidate(task_id="t1", spec=make_spec())
        with pytest.raises(AttributeError):
            tc.task_id = "t2"


# -- AgentCandidate --


class TestAgentCandidate:
    """AgentCandidate 候选 Agent。"""

    def test_has_capacity_default(self):
        from mcc_harness import AgentCandidate
        ac = AgentCandidate(agent_id="a1", max_concurrent=2, current_load=0)
        assert ac.has_capacity is True
        assert ac.remaining_capacity == 2

    def test_no_capacity_when_full(self):
        from mcc_harness import AgentCandidate
        ac = AgentCandidate(agent_id="a1", max_concurrent=2, current_load=2)
        assert ac.has_capacity is False
        assert ac.remaining_capacity == 0

    def test_partial_capacity(self):
        from mcc_harness import AgentCandidate
        ac = AgentCandidate(agent_id="a1", max_concurrent=3, current_load=1)
        assert ac.has_capacity is True
        assert ac.remaining_capacity == 2

    def test_can_serve_with_capabilities(self):
        from mcc_harness import AgentCandidate
        ac = AgentCandidate(
            agent_id="a1",
            capabilities=frozenset({"code", "test"}),
            max_concurrent=2,
        )
        assert ac.can_serve(frozenset({"code"})) is True
        assert ac.can_serve(frozenset({"code", "test"})) is True
        assert ac.can_serve(frozenset({"code", "deploy"})) is False

    def test_cannot_serve_when_full(self):
        from mcc_harness import AgentCandidate
        ac = AgentCandidate(
            agent_id="a1",
            capabilities=frozenset({"code"}),
            max_concurrent=1,
            current_load=1,
        )
        assert ac.can_serve(frozenset({"code"})) is False


# -- TaskSchedulerImpl 基本调度 --


class TestSchedulerBasic:
    """TaskSchedulerImpl 基本调度功能。"""

    async def test_schedule_basic_assignment(self, registry):
        register_agent(registry, "a1")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule([make_spec("task1")], registry)
        assert len(result) == 1
        assert result[0].agent_id == "a1"
        assert result[0].task_spec.subject == "task1"

    async def test_schedule_multiple_tasks_different_agents(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert len(result) == 2
        agent_ids = {a.agent_id for a in result}
        assert agent_ids == {"a1", "a2"}

    async def test_schedule_excludes_agents(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("t1")], registry,
            exclude_agents=frozenset({"a1"}),
        )
        assert len(result) == 1
        assert result[0].agent_id == "a2"

    async def test_schedule_no_agents_returns_empty(self, registry):
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule([make_spec("t1")], registry)
        assert result == []


# -- 依赖 DAG --


class TestDependencyDAG:
    """依赖 DAG：依赖未满足的任务不可调度。"""

    async def test_unsatisfied_deps_deferred(self, registry):
        register_agent(registry, "a1")
        def deps_check(tid):
            return False

        scheduler = TaskSchedulerImpl(dependency_check=deps_check)
        result = await scheduler.schedule_round(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert result.total_assigned == 0
        assert len(result.deferred) == 2

    async def test_satisfied_deps_scheduled(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")

        def deps_check(tid):
            return True

        scheduler = TaskSchedulerImpl(dependency_check=deps_check)
        result = await scheduler.schedule_round(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert result.total_assigned == 2
        assert len(result.deferred) == 0

    async def test_mixed_deps(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        ready_map = {"task_0000": True, "task_0001": False}

        def deps_check(tid):
            return ready_map.get(tid, True)

        scheduler = TaskSchedulerImpl(dependency_check=deps_check)
        result = await scheduler.schedule_round(
            [make_spec("ready"), make_spec("blocked")], registry,
        )
        assert result.total_assigned == 1
        assert len(result.deferred) == 1
        assert result.deferred[0] == "task_0001"

    async def test_task_store_integration(self, registry, task_store):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        t1 = task_store.create(subject="parent", description="d")
        t2 = task_store.create(subject="child", description="d")
        task_store.update_dependencies(t2.id, [t1.id])

        scheduler = TaskSchedulerImpl(
            dependency_check=lambda tid: task_store.can_start(tid),
        )
        result = await scheduler.schedule_round(
            [make_spec("child"), make_spec("parent")],
            registry,
            task_ids=[t2.id, t1.id],
        )
        assert result.total_assigned == 1
        assert len(result.deferred) == 1
        assert result.deferred[0] == t2.id

        parent_agent = result.assignments[0].agent_id
        task_store.claim(t1.id, parent_agent)
        task_store.complete(t1.id, parent_agent)
        scheduler.reset_loads()
        result2 = await scheduler.schedule_round(
            [make_spec("child")], registry,
            task_ids=[t2.id],
        )
        assert result2.total_assigned == 1


# -- 能力匹配 --


class TestCapabilityMatching:
    """能力匹配：仅分配具备所需能力的 Agent。"""

    async def test_capability_match(self, registry):
        register_agent(registry, "a1", frozenset({"python"}))
        register_agent(registry, "a2", frozenset({"rust"}))
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("task", capabilities=frozenset({"python"}))], registry,
        )
        assert len(result) == 1
        assert result[0].agent_id == "a1"

    async def test_capability_no_match_rejected(self, registry):
        register_agent(registry, "a1", frozenset({"rust"}))
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule_round(
            [make_spec("task", capabilities=frozenset({"python"}))], registry,
        )
        assert result.total_assigned == 0
        assert len(result.rejected) == 1

    async def test_multi_capability_match(self, registry):
        register_agent(registry, "a1", frozenset({"python", "test"}))
        register_agent(registry, "a2", frozenset({"python"}))
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("task", capabilities=frozenset({"python", "test"}))], registry,
        )
        assert len(result) == 1
        assert result[0].agent_id == "a1"

    async def test_no_capability_requirement_matches_any(self, registry):
        register_agent(registry, "a1", frozenset({"python"}))
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("task", capabilities=frozenset())], registry,
        )
        assert len(result) == 1


# -- 并发限制 --


class TestConcurrencyLimit:
    """并发上限：单 Agent 最大并发任务数。"""

    async def test_max_concurrent_per_agent(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        policy = SchedulingPolicy(max_concurrent_per_agent=2)
        scheduler = TaskSchedulerImpl(policy)
        result = await scheduler.schedule(
            [make_spec("t1"), make_spec("t2"), make_spec("t3")], registry,
        )
        agent_counts: dict[str, int] = {}
        for a in result:
            agent_counts[a.agent_id] = agent_counts.get(a.agent_id, 0) + 1
        for count in agent_counts.values():
            assert count <= 2

    async def test_concurrent_one_default(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert len(result) == 2
        assert result[0].agent_id != result[1].agent_id

    async def test_max_total_assignments(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        policy = SchedulingPolicy(max_total_assignments=1)
        scheduler = TaskSchedulerImpl(policy)
        result = await scheduler.schedule_round(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert result.total_assigned == 1
        assert len(result.deferred) == 1


# -- 优先级排序 --


class TestPrioritySort:
    """优先级排序：DESC 高优先在前，ASC 低优先在前。"""

    async def test_desc_priority(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        register_agent(registry, "a3")
        scheduler = TaskSchedulerImpl(
            SchedulingPolicy(priority_order=PriorityOrder.DESC),
        )
        specs = [
            make_spec("low", priority=1),
            make_spec("high", priority=10),
            make_spec("mid", priority=5),
        ]
        result = await scheduler.schedule(specs, registry)
        subjects = [a.task_spec.subject for a in result]
        assert subjects[0] == "high"
        assert subjects[1] == "mid"
        assert subjects[2] == "low"

    async def test_asc_priority(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        register_agent(registry, "a3")
        scheduler = TaskSchedulerImpl(
            SchedulingPolicy(priority_order=PriorityOrder.ASC),
        )
        specs = [
            make_spec("low", priority=1),
            make_spec("high", priority=10),
            make_spec("mid", priority=5),
        ]
        result = await scheduler.schedule(specs, registry)
        subjects = [a.task_spec.subject for a in result]
        assert subjects[0] == "low"
        assert subjects[1] == "mid"
        assert subjects[2] == "high"


# -- 同一任务不重复分配 --


class TestNoDuplicateAssignment:
    """同一任务不能重复分配。"""

    async def test_exclude_prevents_reassignment(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        scheduler = TaskSchedulerImpl()
        result1 = await scheduler.schedule(
            [make_spec("t1")], registry,
        )
        used = {a.agent_id for a in result1}
        result2 = await scheduler.schedule(
            [make_spec("t2")], registry,
            exclude_agents=used,
        )
        all_agents = {a.agent_id for a in result1 + result2}
        assert len(all_agents) == 2

    async def test_same_round_no_duplicate_agent(self, registry):
        register_agent(registry, "a1")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("t1"), make_spec("t2")], registry,
        )
        assert len(result) == 1


# -- claim 回调 --


class TestClaimCallback:
    """调度与 claim 原子衔接。"""

    async def test_claim_called_on_assignment(self, registry, task_store):
        register_agent(registry, "a1")
        t1 = task_store.create(subject="task1", description="d")
        claimed: list[tuple[str, str]] = []

        def claim_cb(task_id: str, agent_id: str) -> None:
            task_store.claim(task_id, agent_id)
            claimed.append((task_id, agent_id))

        scheduler = TaskSchedulerImpl(claim_callback=claim_cb)
        result = await scheduler.schedule(
            [make_spec("task1")], registry,
            task_ids=[t1.id],
        )
        assert len(result) == 1
        assert len(claimed) == 1
        assert claimed[0] == (t1.id, "a1")
        task = task_store.load(t1.id)
        assert task.status == "in_progress"

    async def test_claim_failure_does_not_break_scheduling(self, registry):
        register_agent(registry, "a1")

        def bad_claim(task_id: str, agent_id: str) -> None:
            raise RuntimeError("claim failed")

        scheduler = TaskSchedulerImpl(claim_callback=bad_claim)
        result = await scheduler.schedule(
            [make_spec("t1")], registry,
        )
        assert len(result) == 1


# -- RetryHandler --


class TestRetryHandler:
    """失败任务重试处理器。"""

    def test_should_retry_within_limit(self):
        handler = RetryHandler(RetryPolicy(max_retries=3))
        assert handler.should_retry("t1") is True

    def test_should_retry_exhausted(self):
        handler = RetryHandler(RetryPolicy(max_retries=2))
        handler.schedule_retry("t1")
        handler.schedule_retry("t1")
        assert handler.should_retry("t1") is False

    def test_schedule_retry_returns_delay(self):
        rp = RetryPolicy(backoff_secs=2.0, backoff_multiplier=2.0)
        handler = RetryHandler(rp)
        d1 = handler.schedule_retry("t1")
        d2 = handler.schedule_retry("t1")
        assert d1 == 2.0
        assert d2 == 4.0

    def test_schedule_retry_exhausted_raises(self):
        handler = RetryHandler(RetryPolicy(max_retries=1))
        handler.schedule_retry("t1")
        with pytest.raises(RetryExhaustedError):
            handler.schedule_retry("t1")

    def test_retry_callback_invoked(self):
        called: list[tuple[str, float]] = []
        handler = RetryHandler(
            retry_callback=lambda tid, delay: called.append((tid, delay)),
        )
        handler.schedule_retry("t1")
        assert len(called) == 1
        assert called[0][0] == "t1"

    def test_get_retry_count(self):
        handler = RetryHandler()
        assert handler.get_retry_count("t1") == 0
        handler.schedule_retry("t1")
        assert handler.get_retry_count("t1") == 1
        handler.schedule_retry("t1")
        assert handler.get_retry_count("t1") == 2

    def test_reset_single(self):
        handler = RetryHandler()
        handler.schedule_retry("t1")
        handler.reset("t1")
        assert handler.get_retry_count("t1") == 0

    def test_reset_all(self):
        handler = RetryHandler()
        handler.schedule_retry("t1")
        handler.schedule_retry("t2")
        handler.reset_all()
        assert handler.get_retry_count("t1") == 0
        assert handler.get_retry_count("t2") == 0


# -- LeaseManager --


class TestLeaseManager:
    """任务租约管理器。"""

    def test_grant_returns_expiry(self):
        lm = LeaseManager(lease_secs=60.0)
        expires = lm.grant("t1", "a1")
        assert expires > 0
        assert "t1" in lm.active_leases()

    def test_renew_extends_lease(self):
        lm = LeaseManager(lease_secs=60.0)
        lm.grant("t1", "a1")
        import time
        time.sleep(0.01)
        new_expires = lm.renew("t1")
        assert new_expires is not None
        assert new_expires > 0

    def test_renew_nonexistent_returns_none(self):
        lm = LeaseManager()
        assert lm.renew("t1") is None

    def test_revoke_returns_agent_id(self):
        lm = LeaseManager()
        lm.grant("t1", "a1")
        agent_id = lm.revoke("t1")
        assert agent_id == "a1"
        assert "t1" not in lm.active_leases()

    def test_revoke_nonexistent_returns_none(self):
        lm = LeaseManager()
        assert lm.revoke("t1") is None

    def test_is_expired_false_for_active(self):
        lm = LeaseManager(lease_secs=60.0)
        lm.grant("t1", "a1")
        assert lm.is_expired("t1") is False

    def test_is_expired_true_for_past(self):
        lm = LeaseManager(lease_secs=0.01)
        lm.grant("t1", "a1")
        import time
        time.sleep(0.02)
        assert lm.is_expired("t1") is True

    def test_is_expired_false_for_nonexistent(self):
        lm = LeaseManager()
        assert lm.is_expired("t1") is False

    def test_collect_expired(self):
        lm = LeaseManager(lease_secs=0.01)
        lm.grant("t1", "a1")
        lm.grant("t2", "a2")
        import time
        time.sleep(0.02)
        expired = lm.collect_expired()
        assert len(expired) == 2
        expired_tasks = {e[0] for e in expired}
        assert expired_tasks == {"t1", "t2"}

    def test_collect_expired_with_callback(self):
        called: list[str] = []
        lm = LeaseManager(
            lease_secs=0.01,
            timeout_callback=lambda tid: called.append(tid),
        )
        lm.grant("t1", "a1")
        import time
        time.sleep(0.02)
        lm.collect_expired()
        assert "t1" in called

    def test_collect_expired_empty_when_none_expired(self):
        lm = LeaseManager(lease_secs=60.0)
        lm.grant("t1", "a1")
        assert lm.collect_expired() == []

    def test_active_leases(self):
        lm = LeaseManager()
        lm.grant("t1", "a1")
        lm.grant("t2", "a2")
        leases = lm.active_leases()
        assert leases == {"t1": "a1", "t2": "a2"}


# -- Protocol 一致性 --


class TestProtocolConformance:
    """TaskSchedulerImpl 实现 TaskScheduler Protocol。"""

    async def test_implements_protocol(self, registry):
        from mcc_harness import TaskScheduler
        scheduler = TaskSchedulerImpl()
        assert isinstance(scheduler, TaskScheduler)

    async def test_schedule_returns_assignments(self, registry):
        register_agent(registry, "a1")
        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule([make_spec("t1")], registry)
        assert all(isinstance(a, Assignment) for a in result)


# -- 端到端集成 --


class TestEndToEnd:
    """端到端：TaskStore + AgentRegistry + Scheduler。"""

    async def test_full_flow_with_task_store(self, registry, task_store):
        register_agent(registry, "a1", frozenset({"python"}))
        register_agent(registry, "a2", frozenset({"rust"}))

        t1 = task_store.create(subject="py-task", description="d")
        t2 = task_store.create(subject="rs-task", description="d")

        claimed: list[tuple[str, str]] = []

        def claim_cb(task_id: str, agent_id: str) -> None:
            task_store.claim(task_id, agent_id)
            claimed.append((task_id, agent_id))

        scheduler = TaskSchedulerImpl(
            SchedulingPolicy(max_concurrent_per_agent=1),
            claim_callback=claim_cb,
            dependency_check=lambda tid: task_store.can_start(tid),
        )
        specs = [
            make_spec("py-task", capabilities=frozenset({"python"})),
            make_spec("rs-task", capabilities=frozenset({"rust"})),
        ]
        result = await scheduler.schedule_round(
            specs, registry, task_ids=[t1.id, t2.id],
        )
        assert result.total_assigned == 2
        assert len(claimed) == 2
        for tid, aid in claimed:
            task = task_store.load(tid)
            assert task.status == "in_progress"
            assert task.owner == aid

    async def test_retry_flow(self, registry, task_store):
        register_agent(registry, "a1")
        t1 = task_store.create(subject="task", description="d")

        scheduler = TaskSchedulerImpl()
        result = await scheduler.schedule(
            [make_spec("task")], registry, task_ids=[t1.id],
        )
        assert len(result) == 1
        task_store.claim(result[0].task_id, result[0].agent_id)
        task_store.complete(result[0].task_id, result[0].agent_id)
        assert task_store.load(t1.id).status == "completed"

    async def test_dependency_chain(self, registry, task_store):
        register_agent(registry, "a1")
        t1 = task_store.create(subject="a", description="d")
        t2 = task_store.create(subject="b", description="d")
        t3 = task_store.create(subject="c", description="d")
        task_store.update_dependencies(t2.id, [t1.id])
        task_store.update_dependencies(t3.id, [t2.id])

        scheduler = TaskSchedulerImpl(
            dependency_check=lambda tid: task_store.can_start(tid),
            claim_callback=lambda tid, aid: task_store.claim(tid, aid),
        )
        specs = [make_spec("a"), make_spec("b"), make_spec("c")]
        ids = [t1.id, t2.id, t3.id]

        result1 = await scheduler.schedule_round(specs, registry, task_ids=ids)
        assert result1.total_assigned == 1
        assert result1.assignments[0].task_id == t1.id
        assert len(result1.deferred) == 2

        task_store.complete(t1.id, "a1")
        scheduler.reset_loads()
        result2 = await scheduler.schedule_round(
            [make_spec("b"), make_spec("c")], registry,
            task_ids=[t2.id, t3.id],
        )
        assert result2.total_assigned == 1
        assert result2.assignments[0].task_id == t2.id
        assert len(result2.deferred) == 1

        task_store.complete(t2.id, "a1")
        scheduler.reset_loads()
        result3 = await scheduler.schedule_round(
            [make_spec("c")], registry, task_ids=[t3.id],
        )
        assert result3.total_assigned == 1
        assert result3.assignments[0].task_id == t3.id

    async def test_load_balancing(self, registry):
        register_agent(registry, "a1")
        register_agent(registry, "a2")
        policy = SchedulingPolicy(max_concurrent_per_agent=2)
        scheduler = TaskSchedulerImpl(policy)
        result = await scheduler.schedule(
            [make_spec(f"t{i}") for i in range(4)], registry,
        )
        assert len(result) == 4
        counts: dict[str, int] = {}
        for a in result:
            counts[a.agent_id] = counts.get(a.agent_id, 0) + 1
        assert max(counts.values()) <= 2
        assert sum(counts.values()) == 4
