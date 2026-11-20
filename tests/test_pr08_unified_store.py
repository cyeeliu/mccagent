"""PR-08 测试：Orchestrator 统一 TaskStoreProtocol。

P1-06 核心问题：
  TeamOrchestrator 同时持有旧同步 task_store 和 async TaskStoreProtocol（via Dispatcher），
  导致两套写入路径：TaskRecord 和 Attempt 由不同接口、不同写入时机更新。

修复后预期：
  - TeamOrchestrator 接受 async_task_store: TaskStoreProtocol 参数
  - 当 async_task_store 提供时，所有任务操作通过它执行
  - 旧同步 task_store 不再被使用
  - _aggregate_results 从 async store 读取
  - progress / collect_results 从 async store 聚合
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.task_store import InMemoryTaskStore
from mcc_harness import (
    AgentRegistry,
    MessageBus,
    PlanStore,
    TeamOrchestrator,
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
def sync_task_store(tmp_base: Path) -> TaskStore:
    return TaskStore(tmp_base / ".tasks")


@pytest.fixture
def async_task_store() -> InMemoryTaskStore:
    return InMemoryTaskStore()


class FakeDispatcher:
    """测试用 TaskDispatcher 桩。"""

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.wait_count = 0
        self.dispatched_tasks: list[str] = []
        self.fail_attempt_calls: list[tuple[str, str, str]] = []

    async def dispatch(self, *args, **kwargs):
        self.dispatch_count += 1
        from mcc_core.runtime import TaskHandle

        return TaskHandle(
            task_id="fake",
            runtime_id="fake-rt",
            attempt_id="fake-att",
        )

    async def wait_for_result(self, *args, **kwargs):
        self.wait_count += 1
        raise RuntimeError("FakeDispatcher does not support wait_for_result")

    async def fail_attempt(self, *args, **kwargs):
        self.fail_attempt_calls.append(args)


@pytest.fixture
def fake_dispatcher() -> FakeDispatcher:
    return FakeDispatcher()


def _make_orchestrator(
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
    fake_dispatcher: FakeDispatcher,
    *,
    async_task_store=None,
    task_store=None,
) -> TeamOrchestrator:
    return TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        dispatcher=fake_dispatcher,
        task_store=task_store,
        async_task_store=async_task_store,
    )


def _register_agents(orchestrator: TeamOrchestrator, count: int = 2) -> list[str]:
    ids = [f"agent-{i}" for i in range(1, count + 1)]
    for i, aid in enumerate(ids):
        orchestrator.register_agent(
            agent_id=aid,
            agent_name=f"worker-{i + 1}",
            capabilities=frozenset({"code_generation"}),
        )
    return ids


class TestPR08AsyncStoreTaskCreation:
    """async_task_store 提供时，任务通过 async store 创建。"""

    async def test_tasks_created_in_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """start 后任务存在于 async store。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=2)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A\ntask B")

        tasks = await async_task_store.list_tasks()
        assert len(tasks) == 2
        task_ids_in_store = {t.task_id for t in tasks}
        assert task_ids_in_store == run.task_ids

    async def test_sync_store_not_used_when_async_provided(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
        sync_task_store,
    ):
        """async store 提供时，sync store 不被使用。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
            task_store=sync_task_store,
        )
        _register_agents(orch, count=2)
        team = orch.create_team("team-1", "alpha", "lead-1")
        await orch.start(team, "task A\ntask B")

        # sync store 应该为空
        sync_tasks = sync_task_store.list()
        assert len(sync_tasks) == 0


class TestPR08AsyncStoreCompletion:
    """async_task_store 提供时，任务完成通过 async store 更新。"""

    async def test_complete_task_updates_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """complete_task 后 async store 中任务状态为 SUCCEEDED。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=1)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]

        await orch.complete_task(run.run_id, aid, tid, summary="done")

        task = await async_task_store.get_task(tid)
        assert task is not None
        assert task.status is TaskStatus.SUCCEEDED

    async def test_fail_task_updates_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """complete_task with error 后 async store 中任务状态为 FAILED。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=1)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]

        await orch.complete_task(run.run_id, aid, tid, error="boom")

        task = await async_task_store.get_task(tid)
        assert task is not None
        assert task.status is TaskStatus.FAILED


class TestPR08AsyncStoreAggregation:
    """async_task_store 提供时，结果从 async store 聚合。"""

    async def test_collect_results_from_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """collect_results 从 async store 读取终态。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=2)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A\ntask B")
        task_list = sorted(run.task_ids)
        agent_list = sorted(run.agent_ids)

        await orch.complete_task(run.run_id, agent_list[0], task_list[0])
        await orch.complete_task(run.run_id, agent_list[1], task_list[1])

        result = await orch.collect_results(run.run_id)
        assert result.completed_count == 2
        assert result.failed_count == 0

    async def test_progress_from_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """progress 从 async store 读取状态。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=1)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A")
        tid = sorted(run.task_ids)[0]
        aid = sorted(run.agent_ids)[0]

        await orch.complete_task(run.run_id, aid, tid, summary="ok")

        prog = await orch.progress(run.run_id)
        assert prog.completed_tasks == 1
        assert prog.is_terminal


class TestPR08AsyncStoreCancel:
    """async_task_store 提供时，取消通过 async store 更新。"""

    async def test_cancel_updates_async_store(
        self,
        registry,
        bus,
        plan_store,
        fake_dispatcher,
        async_task_store,
    ):
        """cancel 后 async store 中任务状态为 CANCELLED。"""
        orch = _make_orchestrator(
            registry, bus, plan_store, fake_dispatcher,
            async_task_store=async_task_store,
        )
        _register_agents(orch, count=1)
        team = orch.create_team("team-1", "alpha", "lead-1")
        run = await orch.start(team, "task A")

        await orch.cancel(run.run_id, reason="user abort")

        for tid in run.task_ids:
            task = await async_task_store.get_task(tid)
            if task is not None:
                assert task.status is TaskStatus.CANCELLED
