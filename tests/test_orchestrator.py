"""TeamOrchestrator 编排层测试：端到端验证任务拆分、分配、执行、失败和汇总。

P2-06 验收标准：
- 可运行一个 Lead + 多个 Agent 的完整任务流程。
- TeamRun 能查询整体进度和终态。
"""

from __future__ import annotations

import asyncio
import dataclasses
import unittest.mock
from pathlib import Path

import pytest
from mcc_harness import (
    AgentDescriptor,
    AgentRegistry,
    Assignment,
    CapabilityScheduler,
    MessageBus,
    OrchestratorError,
    Planner,
    PlanStore,
    RegistrationStatus,
    ResultCollector,
    RunNotFoundError,
    RunStateError,
    RuntimeCoordinator,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    SimplePlanner,
    TaskScheduler,
    TaskSpec,
    Team,
    TeamNotFoundError,
    TeamOrchestrator,
    TeamPolicy,
    TeamRunResult,
    TeamRunSnapshot,
    TeamRunStatus,
    WaitTimeoutError,
)
from mcc_tools import TaskStore

# -- fixture --


@pytest.fixture
def tmp_base(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry(heartbeat_timeout=30.0)


@pytest.fixture
def bus(tmp_base: Path) -> MessageBus:
    return MessageBus(tmp_base)


@pytest.fixture
def plan_store() -> PlanStore:
    return PlanStore()


@pytest.fixture
def task_store(tmp_base: Path) -> TaskStore:
    return TaskStore(tmp_base / ".tasks")


@pytest.fixture
def orchestrator(
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
) -> TeamOrchestrator:
    return TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
    )


@pytest.fixture
def orchestrator_with_tasks(
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
    task_store: TaskStore,
) -> TeamOrchestrator:
    return TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
    )


def make_team(
    orchestrator: TeamOrchestrator,
    team_id: str = "team-1",
    name: str = "alpha",
    lead_id: str = "lead-1",
) -> Team:
    return orchestrator.create_team(team_id, name, lead_id)


def register_agents(orchestrator: TeamOrchestrator, count: int = 2) -> list[str]:
    ids = [f"agent-{i}" for i in range(1, count + 1)]
    for i, aid in enumerate(ids):
        orchestrator.register_agent(
            agent_id=aid,
            agent_name=f"worker-{i+1}",
            capabilities=frozenset({"code_generation"}),
        )
    return ids


# -- TeamPolicy 校验 --


class TestTeamPolicy:
    """TeamPolicy 参数校验。"""

    def test_default_policy(self):
        policy = TeamPolicy()
        assert policy.max_agents == 4
        assert policy.max_concurrent_tasks == 4
        assert policy.heartbeat_interval_secs > 0
        assert policy.task_lease_secs > 0

    @pytest.mark.parametrize("field,value", [
        ("max_agents", 0),
        ("max_agents", -1),
        ("max_concurrent_tasks", 0),
        ("heartbeat_interval_secs", 0),
        ("heartbeat_interval_secs", -1),
        ("task_lease_secs", 0),
        ("task_lease_secs", -1),
    ])
    def test_invalid_policy_raises(self, field, value):
        with pytest.raises(OrchestratorError):
            TeamPolicy(**{field: value})

    def test_custom_policy(self):
        policy = TeamPolicy(max_agents=8, max_concurrent_tasks=10, require_plan=True)
        assert policy.max_agents == 8
        assert policy.require_plan is True


# -- Team 创建 --


class TestCreateTeam:
    """团队创建与查询。"""

    def test_create_team_basic(self, orchestrator):
        team = make_team(orchestrator)
        assert team.team_id == "team-1"
        assert team.name == "alpha"
        assert team.lead_agent_id == "lead-1"

    def test_create_team_idempotent(self, orchestrator):
        team1 = make_team(orchestrator)
        team2 = make_team(orchestrator)
        assert team1 == team2

    def test_create_team_with_policy(self, orchestrator):
        policy = TeamPolicy(max_agents=2, require_plan=True)
        team = orchestrator.create_team("t", "beta", "lead", policy=policy)
        assert team.policy.max_agents == 2
        assert team.policy.require_plan is True

    def test_create_team_with_tags(self, orchestrator):
        team = orchestrator.create_team(
            "t", "beta", "lead", tags=frozenset({"production", "critical"}),
        )
        assert "production" in team.tags

    def test_get_team_not_found(self, orchestrator):
        with pytest.raises(TeamNotFoundError):
            orchestrator.get_team("nonexistent")

    def test_get_team_exists(self, orchestrator):
        make_team(orchestrator)
        team = orchestrator.get_team("team-1")
        assert team.team_id == "team-1"


# -- start / 生命周期 --


class TestStartRun:
    """TeamRun 启动与基本生命周期。"""

    async def test_start_empty_objective_raises(self, orchestrator):
        team = make_team(orchestrator)
        with pytest.raises(OrchestratorError, match="objective"):
            await orchestrator.start(team, "")

    async def test_start_no_agents(self, orchestrator):
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "do task A")
        assert run.status == TeamRunStatus.RUNNING
        assert len(run.agent_ids) == 0
        assert len(run.task_ids) == 1

    async def test_start_with_agents(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "do task A\ndo task B")
        assert run.status == TeamRunStatus.RUNNING
        assert len(run.task_ids) == 2
        assert len(run.agent_ids) >= 1

    async def test_start_empty_plan_completes(self, orchestrator):
        class EmptyPlanner:
            async def plan(self, objective, *, context=None):
                return []

        orchestrator._planner = EmptyPlanner()
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "objective with no tasks")
        assert run.status == TeamRunStatus.COMPLETED

    async def test_start_custom_run_id(self, orchestrator):
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task", run_id="my-run")
        assert run.run_id == "my-run"

    async def test_start_duplicate_run_id_raises(self, orchestrator):
        team = make_team(orchestrator)
        await orchestrator.start(team, "task", run_id="dup")
        with pytest.raises(RunStateError):
            await orchestrator.start(team, "task", run_id="dup")

    async def test_start_creates_tasks_in_store(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=2)
        team = make_team(orch)
        run = await orch.start(team, "task A\ntask B")
        for tid in run.task_ids:
            if tid:
                task = orch.task_store.load(tid)
                assert task.status == "in_progress"


# -- pause / resume --


class TestPauseResume:
    """暂停与恢复。"""

    async def test_pause_running(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        paused = await orchestrator.pause(run.run_id)
        assert paused.status == TeamRunStatus.PAUSED

    async def test_resume_paused(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        await orchestrator.pause(run.run_id)
        resumed = await orchestrator.resume(run.run_id)
        assert resumed.status == TeamRunStatus.RUNNING

    async def test_pause_not_running_raises(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        await orchestrator.pause(run.run_id)
        with pytest.raises(RunStateError):
            await orchestrator.pause(run.run_id)

    async def test_resume_not_paused_raises(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        with pytest.raises(RunStateError):
            await orchestrator.resume(run.run_id)

    async def test_pause_unknown_run(self, orchestrator):
        with pytest.raises(RunNotFoundError):
            await orchestrator.pause("nonexistent")


# -- cancel / shutdown --


class TestCancelShutdown:
    """取消与关闭。"""

    async def test_cancel_running(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        cancelled = await orchestrator.cancel(run.run_id, reason="user abort")
        assert cancelled.status == TeamRunStatus.CANCELLED
        assert cancelled.error == "user abort"
        assert cancelled.finished_at is not None

    async def test_cancel_already_terminal_raises(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        await orchestrator.cancel(run.run_id)
        with pytest.raises(RunStateError):
            await orchestrator.cancel(run.run_id)

    async def test_shutdown(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        shut = await orchestrator.shutdown(run.run_id)
        assert shut.status == TeamRunStatus.SHUTDOWN
        assert shut.finished_at is not None

    async def test_shutdown_idempotent(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        await orchestrator.shutdown(run.run_id)
        shut = await orchestrator.shutdown(run.run_id)
        assert shut.status == TeamRunStatus.SHUTDOWN

    async def test_cancel_unknown_run(self, orchestrator):
        with pytest.raises(RunNotFoundError):
            await orchestrator.cancel("nonexistent")

    async def test_cancel_cancels_tasks_in_store(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=1)
        team = make_team(orch)
        run = await orch.start(team, "task A\ntask B")
        await orch.cancel(run.run_id)
        for tid in run.task_ids:
            if tid:
                task = orch.task_store.load(tid)
                assert task.status == "cancelled"


# -- complete_task / 结果汇总 --


class TestCompleteTask:
    """任务完成与结果汇总。"""

    async def test_complete_all_tasks(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A\ntask B")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)
        await orchestrator.complete_task(
            run.run_id, agent_list[0], task_list[0], summary="done A",
        )
        assert run.status == TeamRunStatus.RUNNING
        await orchestrator.complete_task(
            run.run_id, agent_list[1], task_list[1], summary="done B",
        )
        assert run.status == TeamRunStatus.COMPLETED
        assert run.finished_at is not None

    async def test_complete_with_failure(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A\ntask B")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)
        await orchestrator.complete_task(
            run.run_id, agent_list[0], task_list[0], summary="done A",
        )
        await orchestrator.complete_task(
            run.run_id, agent_list[1], task_list[1], error="boom",
        )
        assert run.status == TeamRunStatus.FAILED
        assert "failed" in (run.error or "")

    async def test_complete_terminal_run_raises(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        await orchestrator.cancel(run.run_id)
        with pytest.raises(RunStateError):
            await orchestrator.complete_task(run.run_id, "a", "t")

    async def test_complete_updates_task_store(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=1)
        team = make_team(orch)
        run = await orch.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]
        await orch.complete_task(run.run_id, aid, tid, summary="ok")
        task = orch.task_store.load(tid)
        assert task.status == "completed"

    async def test_complete_failure_updates_task_store(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=1)
        team = make_team(orch)
        run = await orch.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]
        await orch.complete_task(run.run_id, aid, tid, error="bad")
        task = orch.task_store.load(tid)
        assert task.status == "failed"


# -- 查询：get_run / list_runs / progress --


class TestQueryRun:
    """TeamRun 查询接口。"""

    async def test_get_run_snapshot(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        snap = orchestrator.get_run(run.run_id)
        assert isinstance(snap, TeamRunSnapshot)
        assert snap.run_id == run.run_id
        assert snap.status == TeamRunStatus.RUNNING
        assert snap.team_id == "team-1"

    async def test_get_run_not_found(self, orchestrator):
        with pytest.raises(RunNotFoundError):
            orchestrator.get_run("nonexistent")

    async def test_list_runs_empty(self, orchestrator):
        assert orchestrator.list_runs() == []

    async def test_list_runs_multiple(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        await orchestrator.start(team, "task A", run_id="r1")
        await orchestrator.start(team, "task B", run_id="r2")
        runs = orchestrator.list_runs()
        assert len(runs) == 2
        ids = {r.run_id for r in runs}
        assert ids == {"r1", "r2"}

    async def test_list_runs_filter_by_team(self, orchestrator):
        register_agents(orchestrator, count=1)
        team_a = orchestrator.create_team("ta", "A", "lead")
        team_b = orchestrator.create_team("tb", "B", "lead")
        await orchestrator.start(team_a, "task", run_id="r1")
        await orchestrator.start(team_b, "task", run_id="r2")
        runs = orchestrator.list_runs(team_id="ta")
        assert len(runs) == 1
        assert runs[0].run_id == "r1"

    async def test_progress_running(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A\ntask B")
        prog = orchestrator.progress(run.run_id)
        assert prog.status == TeamRunStatus.RUNNING
        assert prog.total_tasks == 2
        assert prog.completed_tasks == 0
        assert prog.pending_tasks == 2
        assert prog.active_agents >= 1
        assert prog.elapsed_secs >= 0

    async def test_progress_completed(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]
        await orchestrator.complete_task(run.run_id, aid, tid, summary="ok")
        prog = orchestrator.progress(run.run_id)
        assert prog.completed_tasks == 1
        assert prog.is_terminal

    async def test_snapshot_to_dict(self, orchestrator):
        register_agents(orchestrator, count=1)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A")
        snap = orchestrator.get_run(run.run_id)
        d = snap.to_dict()
        assert d["run_id"] == run.run_id
        assert d["status"] == "running"
        assert isinstance(d["agent_ids"], list)


# -- collect_results --


class TestCollectResults:
    """结果收集。"""

    async def test_collect_all_completed(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A\ntask B")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)
        await orchestrator.complete_task(run.run_id, agent_list[0], task_list[0])
        await orchestrator.complete_task(run.run_id, agent_list[1], task_list[1])
        result = await orchestrator.collect_results(run.run_id)
        assert isinstance(result, TeamRunResult)
        assert result.completed_count == 2
        assert result.failed_count == 0
        assert result.total_count == 2
        assert result.success_rate == 1.0

    async def test_collect_with_failure(self, orchestrator):
        register_agents(orchestrator, count=2)
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "task A\ntask B")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)
        await orchestrator.complete_task(run.run_id, agent_list[0], task_list[0])
        await orchestrator.complete_task(
            run.run_id, agent_list[1], task_list[1], error="fail",
        )
        result = await orchestrator.collect_results(run.run_id)
        assert result.completed_count == 1
        assert result.failed_count == 1
        assert result.success_rate == 0.5

    async def test_collect_unknown_run(self, orchestrator):
        with pytest.raises(RunNotFoundError):
            await orchestrator.collect_results("nonexistent")


# -- 端到端 --


class TestEndToEnd:
    """端到端：Lead + 多 Agent 完整流程。"""

    async def test_full_flow_success(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        agent_ids = register_agents(orch, count=3)
        team = make_team(orch)
        run = await orch.start(team, "实现模块A\n实现模块B\n实现模块C")
        assert run.status == TeamRunStatus.RUNNING
        assert len(run.task_ids) == 3
        assert len(run.agent_ids) >= 1
        task_list = sorted(run.task_ids)
        for i, tid in enumerate(task_list):
            aid = agent_ids[i % len(agent_ids)]
            await orch.complete_task(
                run.run_id, aid, tid, summary=f"模块{i+1}完成",
            )
        assert run.status == TeamRunStatus.COMPLETED
        prog = orch.progress(run.run_id)
        assert prog.completed_tasks == 3
        assert prog.failed_tasks == 0
        result = await orch.collect_results(run.run_id)
        assert result.success_rate == 1.0

    async def test_full_flow_partial_failure(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        agent_ids = register_agents(orch, count=2)
        team = make_team(orch)
        run = await orch.start(team, "任务一\n任务二")
        task_list = sorted(run.task_ids)
        await orch.complete_task(
            run.run_id, agent_ids[0], task_list[0], summary="ok",
        )
        await orch.complete_task(
            run.run_id, agent_ids[1], task_list[1], error="执行失败",
        )
        assert run.status == TeamRunStatus.FAILED
        prog = orch.progress(run.run_id)
        assert prog.completed_tasks == 1
        assert prog.failed_tasks == 1

    async def test_full_flow_cancel_midway(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=2)
        team = make_team(orch)
        run = await orch.start(team, "任务一\n任务二\n任务三")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)
        await orch.complete_task(run.run_id, agent_list[0], task_list[0])
        await orch.cancel(run.run_id, reason="用户取消")
        assert run.status == TeamRunStatus.CANCELLED
        for tid in task_list[1:]:
            task = orch.task_store.load(tid)
            assert task.status == "cancelled"

    async def test_full_flow_pause_resume(self, orchestrator_with_tasks):
        orch = orchestrator_with_tasks
        register_agents(orch, count=1)
        team = make_team(orch)
        run = await orch.start(team, "任务一")
        await orch.pause(run.run_id)
        assert orch.get_run(run.run_id).status == TeamRunStatus.PAUSED
        await orch.resume(run.run_id)
        assert orch.get_run(run.run_id).status == TeamRunStatus.RUNNING
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]
        await orch.complete_task(run.run_id, aid, tid, summary="done")
        assert run.status == TeamRunStatus.COMPLETED


# -- 自定义 Planner / Scheduler --


class TestCustomPlanner:
    """自定义 Planner 注入。"""

    async def test_custom_planner(self, orchestrator):
        class TwoTaskPlanner:
            async def plan(self, objective, *, context=None):
                return [
                    TaskSpec(subject=f"{objective}-part1"),
                    TaskSpec(subject=f"{objective}-part2"),
                ]

        register_agents(orchestrator, count=2)
        orchestrator._planner = TwoTaskPlanner()
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "build")
        assert len(run.task_ids) == 2


class TestCustomScheduler:
    """自定义 Scheduler 注入。"""

    async def test_custom_scheduler(self, orchestrator):
        class RoundRobinScheduler:
            async def schedule(self, tasks, registry, *, exclude_agents=frozenset()):
                agents = registry.list(status=RegistrationStatus.AVAILABLE)
                assignments = []
                for i, spec in enumerate(tasks):
                    if agents:
                        aid = agents[i % len(agents)].agent_id
                        assignments.append(Assignment(task_id="", agent_id=aid, task_spec=spec))
                return assignments

        register_agents(orchestrator, count=3)
        orchestrator._scheduler = RoundRobinScheduler()
        team = make_team(orchestrator)
        run = await orchestrator.start(team, "t1\nt2\nt3")
        assert len(run.agent_ids) == 3


# -- 默认实现 --


class TestSimplePlanner:
    """SimplePlanner 默认实现。"""

    async def test_plan_multiline(self):
        planner = SimplePlanner()
        specs = await planner.plan("任务A\n任务B\n任务C")
        assert len(specs) == 3
        assert specs[0].subject == "任务A"
        assert specs[0].priority > specs[1].priority

    async def test_plan_empty_lines(self):
        planner = SimplePlanner()
        specs = await planner.plan("任务A\n\n\n任务B")
        assert len(specs) == 2

    async def test_plan_whitespace_only(self):
        planner = SimplePlanner()
        specs = await planner.plan("   \n  \n")
        assert specs == []


class TestCapabilityScheduler:
    """CapabilityScheduler 默认实现。"""

    async def test_schedule_by_capability(self, registry):
        registry.register(AgentDescriptor(
            agent_id="a1",
            agent_name="coder",
            capabilities=frozenset({"code_generation"}),
        ))
        registry.register(AgentDescriptor(
            agent_id="a2",
            agent_name="reviewer",
            capabilities=frozenset({"code_review"}),
        ))
        scheduler = CapabilityScheduler()
        specs = [TaskSpec(subject="write", required_capabilities=frozenset({"code_generation"}))]
        assignments = await scheduler.schedule(specs, registry)
        assert len(assignments) == 1
        assert assignments[0].agent_id == "a1"

    async def test_schedule_no_capability_any_agent(self, registry):
        registry.register(AgentDescriptor(
            agent_id="a1", agent_name="worker", capabilities=frozenset(),
        ))
        scheduler = CapabilityScheduler()
        specs = [TaskSpec(subject="generic")]
        assignments = await scheduler.schedule(specs, registry)
        assert len(assignments) == 1

    async def test_schedule_no_agents_empty(self, registry):
        scheduler = CapabilityScheduler()
        specs = [TaskSpec(subject="orphan")]
        assignments = await scheduler.schedule(specs, registry)
        assert assignments == []

    async def test_schedule_exclude_agents(self, registry):
        registry.register(AgentDescriptor(
            agent_id="a1", agent_name="w1", capabilities=frozenset({"code"}),
        ))
        registry.register(AgentDescriptor(
            agent_id="a2", agent_name="w2", capabilities=frozenset({"code"}),
        ))
        scheduler = CapabilityScheduler()
        specs = [TaskSpec(subject="task", required_capabilities=frozenset({"code"}))]
        assignments = await scheduler.schedule(specs, registry, exclude_agents=frozenset({"a1"}))
        assert assignments[0].agent_id == "a2"


# -- Protocol 检查 --


class TestProtocolConformance:
    """Protocol 接口一致性。"""

    def test_simple_planner_is_planner(self):
        planner = SimplePlanner()
        assert isinstance(planner, Planner)

    def test_capability_scheduler_is_scheduler(self):
        scheduler = CapabilityScheduler()
        assert isinstance(scheduler, TaskScheduler)

    def test_simple_result_collector_is_collector(self, orchestrator):
        from mcc_harness import SimpleResultCollector
        collector = SimpleResultCollector(orchestrator)
        assert isinstance(collector, ResultCollector)


# -- register_agent 便捷方法 --


class TestRegisterAgent:
    """register_agent 便捷方法。"""

    def test_register_agent_basic(self, orchestrator):
        reg = orchestrator.register_agent(
            "a1", "worker-1",
            capabilities=frozenset({"code"}),
        )
        assert reg.agent_id == "a1"
        assert reg.is_available

    def test_register_agent_with_mode(self, orchestrator):
        reg = orchestrator.register_agent(
            "a1", "ephemeral",
            mode=RuntimeMode.EPHEMERAL,
        )
        assert reg.descriptor.mode == RuntimeMode.EPHEMERAL

    def test_register_agent_in_registry(self, orchestrator):
        orchestrator.register_agent(
            "a1", "worker",
            capabilities=frozenset({"code_generation"}),
        )
        found = orchestrator.registry.find_by_capability("code_generation")
        assert len(found) == 1


# -- 执行闭环（Phase 5）--


class StubRuntime:
    """测试用 Runtime 桩：实现 AgentRuntime + PausableRuntime 协议。

    通过预设 result 控制执行结果，通过 call_log 记录调用顺序。
    """

    def __init__(
        self,
        runtime_id: str,
        result: RuntimeResult | None = None,
        raise_timeout: bool = False,
        raise_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._runtime_id = runtime_id
        self._status = RuntimeStatus.CREATED
        self._result = result or RuntimeResult(
            runtime_id=runtime_id,
            status=RuntimeStatus.COMPLETED,
            summary="done",
        )
        self._raise_timeout = raise_timeout
        self._raise_error = raise_error
        self._delay = delay
        self.call_log: list[str] = []
        self.start_count = 0
        self.cancelled = False
        self.paused = False
        self.resumed = False
        self.stopped = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status not in (
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
            RuntimeStatus.TIMEOUT,
            RuntimeStatus.SHUTDOWN,
        )

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        self.call_log.append(f"start:{task}")
        self.start_count += 1
        self._status = RuntimeStatus.RUNNING

    async def stop(self, *, timeout: float = 5.0) -> None:
        self.call_log.append("stop")
        self._status = RuntimeStatus.SHUTDOWN
        self.stopped = True

    def cancel(self) -> None:
        self.call_log.append("cancel")
        self.cancelled = True
        self._status = RuntimeStatus.CANCELLED

    def pause(self) -> None:
        self.call_log.append("pause")
        self.paused = True
        self._status = RuntimeStatus.PAUSED

    def resume(self) -> None:
        self.call_log.append("resume")
        self.resumed = True
        self._status = RuntimeStatus.RUNNING

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_done")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise_error is not None:
            raise self._raise_error
        if self._raise_timeout:
            raise WaitTimeoutError("rt-stub", 1.0)
        self._status = self._result.status
        return self._result

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        self.call_log.append(f"submit_task:{task}")
        self.start_count += 1
        self._status = RuntimeStatus.RUNNING
        return task_id or f"task::{task}"

    async def wait_task(self, task_id: str, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_task")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise_error is not None:
            raise self._raise_error
        if self._raise_timeout:
            raise WaitTimeoutError("rt-stub", 1.0)
        self._status = self._result.status
        return self._result


@pytest.fixture
def coordinator() -> RuntimeCoordinator:
    return RuntimeCoordinator()


@pytest.fixture
def orchestrator_with_coordinator(
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
    task_store: TaskStore,
    coordinator: RuntimeCoordinator,
) -> TeamOrchestrator:
    return TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        coordinator=coordinator,
    )


class TestOrchestratorExecutionLoop:
    """execute_run 执行闭环：Runtime 执行 → 结果回流 → 状态更新。"""

    async def test_execute_run_completes_all_assignments(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 2)
        team = make_team(orch)

        rt1 = StubRuntime("rt-1", RuntimeResult(
            runtime_id="rt-1", status=RuntimeStatus.COMPLETED, summary="ok-1",
        ))
        rt2 = StubRuntime("rt-2", RuntimeResult(
            runtime_id="rt-2", status=RuntimeStatus.COMPLETED, summary="ok-2",
        ))
        coordinator.register(rt1)
        coordinator.register(rt2)
        orch.bind_runtime(agent_ids[0], "rt-1")
        orch.bind_runtime(agent_ids[1], "rt-2")

        run = await orch.start(team, "build feature A\nbuild feature B")
        run = await orch.execute_run(run.run_id)

        assert run.status == TeamRunStatus.COMPLETED
        assert len(run.results) == 2
        summaries = {r.summary for r in run.results}
        assert summaries == {"ok-1", "ok-2"}

    async def test_execute_run_no_runtime_fails(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
    ):
        orch = orchestrator_with_coordinator
        register_agents(orch, 1)
        team = make_team(orch)

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        assert run.status == TeamRunStatus.FAILED
        assert len(run.results) == 1
        assert run.results[0].status == "failed"
        assert "no runtime bound" in (run.results[0].error or "")

    async def test_execute_run_partial_failure(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 2)
        team = make_team(orch)

        rt1 = StubRuntime("rt-1", RuntimeResult(
            runtime_id="rt-1", status=RuntimeStatus.COMPLETED, summary="ok",
        ))
        rt2 = StubRuntime("rt-2", RuntimeResult(
            runtime_id="rt-2", status=RuntimeStatus.FAILED, summary="",
            error="boom",
        ))
        coordinator.register(rt1)
        coordinator.register(rt2)
        orch.bind_runtime(agent_ids[0], "rt-1")
        orch.bind_runtime(agent_ids[1], "rt-2")

        run = await orch.start(team, "build feature A\nbuild feature B")
        run = await orch.execute_run(run.run_id)

        assert run.status == TeamRunStatus.FAILED
        statuses = {r.status for r in run.results}
        assert "completed" in statuses
        assert "failed" in statuses

    async def test_execute_run_concurrency_limit(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        policy = TeamPolicy(max_concurrent_tasks=1)
        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 2)
        team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

        execution_log: list[tuple[str, float]] = []

        class TimedStubRuntime(StubRuntime):
            async def start(self, task: str = "", *, task_id: str | None = None) -> None:
                await super().start(task, task_id=task_id)
                execution_log.append((self._runtime_id, asyncio.get_event_loop().time()))

            async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
                await asyncio.sleep(0.05)
                self._status = self._result.status
                return self._result

            async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
                await self.start(task, task_id=task_id)
                return task_id or f"task::{task}"

            async def wait_task(
                self, task_id: str, *, timeout: float | None = None,
            ) -> RuntimeResult:
                return await self.wait_done(timeout=timeout)

        rt1 = TimedStubRuntime("rt-1")
        rt2 = TimedStubRuntime("rt-2")
        coordinator.register(rt1)
        coordinator.register(rt2)
        orch.bind_runtime(agent_ids[0], "rt-1")
        orch.bind_runtime(agent_ids[1], "rt-2")

        run = await orch.start(team, "build feature A\nbuild feature B")
        run = await orch.execute_run(run.run_id)

        assert run.status == TeamRunStatus.COMPLETED
        assert len(execution_log) == 2
        times = sorted(t for _, t in execution_log)
        assert times[1] - times[0] >= 0.04

    async def test_execute_run_retry_on_failure(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        policy = TeamPolicy(task_lease_secs=10.0)
        policy_with_meta = dataclasses.replace(policy)
        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = orch.create_team("team-1", "alpha", "lead-1", policy=policy_with_meta)

        call_count = 0

        class RetryStubRuntime(StubRuntime):
            async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return RuntimeResult(
                        runtime_id=self._runtime_id,
                        status=RuntimeStatus.FAILED,
                        summary="",
                        error=f"fail-{call_count}",
                    )
                return RuntimeResult(
                    runtime_id=self._runtime_id,
                    status=RuntimeStatus.COMPLETED,
                    summary="ok",
                )

            async def wait_task(
                self, task_id: str, *, timeout: float | None = None,
            ) -> RuntimeResult:
                return await self.wait_done(timeout=timeout)

        rt = RetryStubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        original_execute = orch._execute_assignment

        async def patched_execute(run_id, assignment, semaphore, max_attempts=1, **kwargs):
            return await original_execute(run_id, assignment, semaphore, max_attempts=3, **kwargs)

        with unittest.mock.patch.object(orch, "_execute_assignment", patched_execute):
            run = await orch.execute_run(run.run_id)

        assert call_count == 3
        assert run.status == TeamRunStatus.COMPLETED

    async def test_execute_run_timeout(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubRuntime("rt-1", raise_timeout=True)
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        assert run.status == TeamRunStatus.FAILED
        assert run.results[0].status == "failed"
        assert "timed out" in (run.results[0].error or "")

    async def test_cancel_propagates_to_runtime(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        rt._status = RuntimeStatus.RUNNING
        await orch.cancel(run.run_id, reason="user abort")

        assert rt.cancelled
        assert run.status == TeamRunStatus.CANCELLED

    async def test_pause_resume_propagates_to_runtime(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        rt._status = RuntimeStatus.RUNNING
        await orch.pause(run.run_id)
        assert rt.paused

        await orch.resume(run.run_id)
        assert rt.resumed

    async def test_shutdown_propagates_to_runtime(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        rt._status = RuntimeStatus.RUNNING
        await orch.shutdown(run.run_id)

        assert rt.stopped
        assert run.status == TeamRunStatus.SHUTDOWN

    def test_bind_unbind_runtime(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
        coordinator: RuntimeCoordinator,
    ):
        orch = orchestrator_with_coordinator
        assert orch.coordinator is not None

        rt = StubRuntime("rt-x")
        coordinator.register(rt)
        orch.bind_runtime("agent-x", "rt-x")
        runtime = orch._find_runtime("agent-x")
        assert runtime is not None
        assert runtime.runtime_id == "rt-x"

        orch.unbind_runtime("agent-x")
        assert orch._find_runtime("agent-x") is None

    def test_find_runtime_no_coordinator(self, orchestrator_with_tasks: TeamOrchestrator):
        assert orchestrator_with_tasks.coordinator is None
        assert orchestrator_with_tasks._find_runtime("any") is None

    @pytest.mark.parametrize("status,has_error", [
        (RuntimeStatus.COMPLETED, False),
        (RuntimeStatus.SHUTDOWN, False),
        (RuntimeStatus.FAILED, True),
        (RuntimeStatus.CANCELLED, True),
        (RuntimeStatus.TIMEOUT, True),
        (RuntimeStatus.BUDGET_EXCEEDED, True),
    ])
    def test_extract_error(self, status: RuntimeStatus, has_error: bool):
        result = RuntimeResult(
            runtime_id="rt", status=status, summary="",
            error="err" if has_error else None,
        )
        error = TeamOrchestrator._extract_error(result)
        if has_error:
            assert error is not None
        else:
            assert error is None

    def test_extract_error_falls_back_to_status(self):
        result = RuntimeResult(
            runtime_id="rt", status=RuntimeStatus.CANCELLED, summary="",
            error=None,
        )
        error = TeamOrchestrator._extract_error(result)
        assert error == RuntimeStatus.CANCELLED.value

    async def test_execute_run_empty_assignments(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        class EmptyPlanner:
            async def plan(self, objective, *, context=None):
                return []

        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
            planner=EmptyPlanner(),
        )
        team = orch.create_team("team-1", "alpha", "lead-1")

        run = await orch.start(team, "do nothing")
        assert run.status == TeamRunStatus.COMPLETED
        assert len(run.assignments) == 0
        with pytest.raises(RunStateError):
            await orch.execute_run(run.run_id)

    async def test_execute_run_wrong_state_raises(
        self,
        orchestrator_with_coordinator: TeamOrchestrator,
    ):
        orch = orchestrator_with_coordinator
        register_agents(orch, 1)
        team = make_team(orch)

        run = await orch.start(team, "build feature A")
        await orch.cancel(run.run_id)

        with pytest.raises(RunStateError):
            await orch.execute_run(run.run_id)


# -- P1-02: PersistentRuntime submit_task/wait_task 执行路径 --


class StubPersistentRuntime:
    """模拟常驻运行时：有 submit_task/wait_task，不走 start/wait_done。

    P1-02 核心验证：execute_run 对常驻 Runtime 用 submit_task + wait_task，
    不调 start（避免 already running）也不调 wait_done（避免等整个循环）。
    """

    def __init__(
        self,
        runtime_id: str,
        result: RuntimeResult | None = None,
    ) -> None:
        self._runtime_id = runtime_id
        self._status = RuntimeStatus.IDLE
        self._result = result or RuntimeResult(
            runtime_id=runtime_id,
            status=RuntimeStatus.COMPLETED,
            summary="done",
        )
        self.submit_count = 0
        self.wait_task_count = 0
        self.start_count = 0
        self.wait_done_count = 0
        self.submitted_tasks: list[str] = []

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return True

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        self.start_count += 1

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        self.wait_done_count += 1
        return self._result

    async def stop(self, *, timeout: float = 5.0) -> None:
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        self._status = RuntimeStatus.CANCELLED

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        self.submit_count += 1
        tid = task_id or f"task-{self.submit_count}"
        self.submitted_tasks.append(tid)
        return tid

    async def wait_task(
        self, task_id: str, *, timeout: float | None = None,
    ) -> RuntimeResult:
        self.wait_task_count += 1
        return self._result


class TestOrchestratorPersistentExecution:
    """P1-02: execute_run 对 PersistentRuntime 走 submit_task + wait_task 路径。"""

    async def test_persistent_runtime_uses_submit_wait_task(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """execute_run 对有 submit_task 的 Runtime 调 submit_task + wait_task。"""
        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubPersistentRuntime("rt-1", RuntimeResult(
            runtime_id="rt-1", status=RuntimeStatus.COMPLETED, summary="ok",
        ))
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        assert rt.submit_count == 1
        assert rt.wait_task_count == 1
        assert rt.start_count == 0
        assert rt.wait_done_count == 0
        assert run.status == TeamRunStatus.COMPLETED
        assert run.results[0].status == "completed"
        assert run.results[0].summary == "ok"

    async def test_persistent_runtime_multiple_tasks_no_already_running(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """同一 PersistentRuntime 处理多个任务不触发 already running 错误。"""
        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 2)
        team = make_team(orch)

        rt = StubPersistentRuntime("rt-1", RuntimeResult(
            runtime_id="rt-1", status=RuntimeStatus.COMPLETED, summary="ok",
        ))
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")
        orch.bind_runtime(agent_ids[1], "rt-1")

        run = await orch.start(team, "build feature A\nbuild feature B")
        run = await orch.execute_run(run.run_id)

        assert rt.submit_count == 2
        assert rt.wait_task_count == 2
        assert rt.start_count == 0
        assert run.status == TeamRunStatus.COMPLETED

    async def test_persistent_runtime_failure_mapped(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """PersistentRuntime 任务失败正确映射到 AgentResult。"""
        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubPersistentRuntime("rt-1", RuntimeResult(
            runtime_id="rt-1", status=RuntimeStatus.FAILED, summary="",
            error="boom",
        ))
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        assert rt.submit_count == 1
        assert rt.wait_task_count == 1
        assert run.status == TeamRunStatus.FAILED
        assert run.results[0].status == "failed"

    async def test_persistent_runtime_timeout_raises_wait_timeout(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """PersistentRuntime wait_task 超时映射到任务失败。"""

        class TimeoutPersistentRuntime(StubPersistentRuntime):
            async def wait_task(
                self, task_id: str, *, timeout: float | None = None,
            ) -> RuntimeResult:
                self.wait_task_count += 1
                raise WaitTimeoutError("rt-stub", 1.0)

        orch = TeamOrchestrator(
            registry=registry, bus=bus, plan_store=plan_store,
            task_store=task_store, coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = TimeoutPersistentRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        assert rt.submit_count == 1
        assert rt.wait_task_count == 1
        assert run.status == TeamRunStatus.FAILED
        assert "timed out" in (run.results[0].error or "")
