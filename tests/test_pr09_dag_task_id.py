"""PR-09: DAG 依赖字典以 task_id 为键，避免 duplicate subject 碰撞。

旧实现中 execute_run 构建 subject_to_assignment / completion_events /
subject_outcomes 三个字典时以 task_spec.subject 为键。当两个 TaskSpec
的 subject 相同时，后者的字典项覆盖前者，导致：

1. completion_events 只保留一个 Event，依赖任务只等待其中一个完成。
2. subject_outcomes 被后完成的任务覆盖，依赖策略检查看到的是"最后写入"
   的结果而非全部依赖任务的结果。
3. 依赖任务可能在部分同名任务尚未完成时就执行，或在全部同名任务都成功时
   因最后写入的是 failed 而被错误跳过。

修复后将三个字典改为以 task_id 为键，blocked_by 中的 subject 通过
subject_to_task_ids 映射展开为全部依赖 task_id，确保每个任务独立追踪。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core.runtime import RuntimeEvent, RuntimeEventType
from mcc_harness import (
    AgentRegistry,
    DependencyPolicy,
    MessageBus,
    PlanStore,
    RuntimeCoordinator,
    RuntimeResult,
    RuntimeStatus,
    TaskSpec,
    TeamOrchestrator,
    TeamRunStatus,
)
from mcc_tools import TaskStore

# 复用 test_orchestrator.py 中的测试辅助类
from tests.test_orchestrator import (
    FakeDispatcher,
    StubRuntime,
    _DagPlanner,
    make_team,
    register_agents,
)

# -- fixtures --


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
def coordinator() -> RuntimeCoordinator:
    return RuntimeCoordinator()


# -- 辅助：支持按 runtime_id 设置延迟的 FakeDispatcher --


class _DelayedDispatcher(FakeDispatcher):
    """FakeDispatcher 子类：支持按 runtime_id 设置独立延迟。

    用于控制同名 subject 任务的完成顺序，使测试确定性可复现。
    """

    def __init__(
        self,
        *,
        runtime_delays: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._runtime_delays = runtime_delays or {}

    async def wait_for_result(self, handle, *, timeout=None) -> RuntimeEvent:
        delay = self._runtime_delays.get(handle.runtime_id, 0.0)
        if delay > 0:
            import asyncio

            await asyncio.sleep(delay)
        return await super().wait_for_result(handle, timeout=timeout)


def _build_orch(
    *,
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
    task_store: TaskStore,
    coordinator: RuntimeCoordinator,
    dispatcher: FakeDispatcher,
    specs: list[TaskSpec],
    policy=None,
) -> tuple[TeamOrchestrator, object, list[str]]:
    """构建 orchestrator + 3 agent + 3 runtime 的测试拓扑。"""
    from mcc_harness import Team, TeamPolicy

    orch = TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        coordinator=coordinator,
        dispatcher=dispatcher,
        planner=_DagPlanner(specs),
    )
    agent_ids = register_agents(orch, 3)
    if policy is not None:
        team = orch.create_team(
            "team-1", "alpha", "lead-1",
            policy=TeamPolicy(dependency_policy=policy),
        )
    else:
        team = make_team(orch)
    for i, rt_id in enumerate(("rt-1", "rt-2", "rt-3")):
        coordinator.register(StubRuntime(rt_id))
        orch.bind_runtime(agent_ids[i], rt_id)
    return orch, team, agent_ids


# -- 测试 --


class TestPr09DagTaskIdKey:
    """PR-09: duplicate subject 不碰撞，DAG 以 task_id 为键。"""

    async def test_duplicate_subject_both_succeed_dependent_executes(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """两个相同 subject 的任务都成功 → 依赖任务正常执行。"""
        dispatcher = FakeDispatcher()
        dispatcher.set_success("rt-1", "build-a-ok")
        dispatcher.set_success("rt-2", "build-b-ok")
        dispatcher.set_success("rt-3", "deploy-ok")

        specs = [
            TaskSpec(subject="build", description="build-frontend"),
            TaskSpec(subject="build", description="build-backend"),
            TaskSpec(subject="deploy", description="deploy", blocked_by=("build",)),
        ]
        orch, team, _ = _build_orch(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            coordinator=coordinator,
            dispatcher=dispatcher,
            specs=specs,
        )

        run = await orch.start(team, "build\ndeploy")
        run = await orch.execute_run(run.run_id)

        # 三个任务都应被派发
        assert dispatcher.dispatch_count == 3
        # run 应到达终态
        assert run.status == TeamRunStatus.COMPLETED
        # 三个结果都应为 completed
        assert len(run.results) == 3
        assert all(r.status == "completed" for r in run.results)

    async def test_duplicate_subject_one_fails_dependent_skipped(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """同名 subject 一个成功一个失败 → ALL_SUCCESS 下依赖任务跳过。

        旧实现以 subject 为键，completion_events 只保留一个 Event。
        rt-1（成功）先完成时设置该 Event 并写入 subject_outcomes["build"]
        ="succeeded"，依赖任务立刻唤醒并通过 ALL_SUCCESS 检查，在 rt-2
        （失败）尚未完成时就执行——这是 bug。

        修复后以 task_id 为键，依赖任务等待全部同名任务的 Event，rt-2
        完成后 dep_outcomes=["succeeded","failed"]，ALL_SUCCESS 不满足，
        依赖任务正确跳过。
        """
        dispatcher = _DelayedDispatcher(
            runtime_delays={"rt-2": 0.05},
        )
        dispatcher.set_success("rt-1", "build-a-ok")
        dispatcher.set_failure("rt-2", "build-b-fail")
        dispatcher.set_success("rt-3", "deploy-ok")

        specs = [
            TaskSpec(subject="build", description="build-frontend"),
            TaskSpec(subject="build", description="build-backend"),
            TaskSpec(subject="deploy", description="deploy", blocked_by=("build",)),
        ]
        orch, team, _ = _build_orch(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            coordinator=coordinator,
            dispatcher=dispatcher,
            specs=specs,
        )

        run = await orch.start(team, "build\ndeploy")
        run = await orch.execute_run(run.run_id)

        # deploy 不应被派发（ALL_SUCCESS：一个 build 失败 → 跳过）
        assert dispatcher.dispatch_count == 2, (
            f"deploy should be skipped, got dispatch_count={dispatcher.dispatch_count}"
        )
        # deploy 结果应为 failed（skipped）
        deploy_result = next(
            r for r in run.results if "skipped" in (r.error or "")
        )
        assert deploy_result.status == "failed"

    async def test_duplicate_subject_all_terminal_policy_executes(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """同名 subject 一个成功一个失败 → ALL_TERMINAL 下依赖任务执行。

        ALL_TERMINAL 只要求依赖到达任意终态，不要求全部成功。
        """
        dispatcher = _DelayedDispatcher(
            runtime_delays={"rt-2": 0.05},
        )
        dispatcher.set_success("rt-1", "build-a-ok")
        dispatcher.set_failure("rt-2", "build-b-fail")
        dispatcher.set_success("rt-3", "deploy-ok")

        specs = [
            TaskSpec(subject="build", description="build-frontend"),
            TaskSpec(subject="build", description="build-backend"),
            TaskSpec(subject="deploy", description="deploy", blocked_by=("build",)),
        ]
        orch, team, _ = _build_orch(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            coordinator=coordinator,
            dispatcher=dispatcher,
            specs=specs,
            policy=DependencyPolicy.ALL_TERMINAL,
        )

        run = await orch.start(team, "build\ndeploy")
        run = await orch.execute_run(run.run_id)

        # ALL_TERMINAL：两个 build 都到达终态 → deploy 执行
        assert dispatcher.dispatch_count == 3, (
            f"deploy should execute under ALL_TERMINAL, "
            f"got dispatch_count={dispatcher.dispatch_count}"
        )

    async def test_duplicate_subject_outcomes_tracked_independently(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """两个同名 subject 的任务结果独立追踪，不互相覆盖。"""
        dispatcher = FakeDispatcher()
        dispatcher.set_success("rt-1", "build-a-ok")
        dispatcher.set_failure("rt-2", "build-b-fail")
        dispatcher.set_success("rt-3", "deploy-ok")

        specs = [
            TaskSpec(subject="build", description="build-frontend"),
            TaskSpec(subject="build", description="build-backend"),
            TaskSpec(subject="deploy", description="deploy", blocked_by=("build",)),
        ]
        orch, team, _ = _build_orch(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            coordinator=coordinator,
            dispatcher=dispatcher,
            specs=specs,
            policy=DependencyPolicy.ALL_TERMINAL,
        )

        run = await orch.start(team, "build\ndeploy")
        run = await orch.execute_run(run.run_id)

        # 应有 3 个结果：1 completed (build-a) + 1 failed (build-b) + 1 completed (deploy)
        assert len(run.results) == 3
        statuses = sorted(r.status for r in run.results)
        assert statuses == ["completed", "completed", "failed"], (
            f"expected [completed, completed, failed], got {statuses}"
        )
