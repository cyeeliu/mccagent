"""Phase 7 深度修复测试：状态副本消除（P1-11）。

验证 TeamRun.results 不再作为独立可写状态副本，
所有结果读取统一通过 _aggregate_results 从 TaskStore 聚合。

关键修复点：
- SimpleResultCollector.collect 通过 _aggregate_results 读取，而非直接读 run.results。
- _check_run_completion 通过 _aggregate_results 判定终态，而非直接读 run.results。
- 非终态 task（pending/claimed/in_progress）不参与终态判定，避免提前误判。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from mcc_harness import (
    AgentRegistry,
    MessageBus,
    PlanStore,
    SimpleResultCollector,
    TaskSpec,
    TeamOrchestrator,
    TeamRunStatus,
)
from mcc_tools import TaskStore
from mcc_tools.task import TaskStatus

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


class _NoOpDispatcher:
    """Phase 7 测试不需要 dispatch/wait，只需一个满足协议的桩。"""

    async def dispatch(
        self,
        task_id: str,
        agent_id: str,
        *,
        runtime_id: str,
        objective: str | None = None,
        input_data: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ):
        from mcc_core.runtime import TaskHandle

        return TaskHandle(task_id=task_id, attempt_id=f"att-{task_id}", runtime_id=runtime_id)

    async def wait_for_result(self, handle, *, timeout=None):
        raise RuntimeError("not used in Phase 7 tests")

    async def fail_attempt(self, handle, *, error, status="failed"):
        pass


@pytest.fixture
def dispatcher() -> _NoOpDispatcher:
    return _NoOpDispatcher()


@pytest.fixture
def orchestrator(
    registry: AgentRegistry,
    bus: MessageBus,
    plan_store: PlanStore,
    task_store: TaskStore,
    dispatcher: _NoOpDispatcher,
) -> TeamOrchestrator:
    return TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        dispatcher=dispatcher,
    )


async def _setup_run(orch: TeamOrchestrator, n_tasks: int = 2):
    """创建 team + 注册 agent + start_from_plan，返回 (run, task_ids, agent_map)。

    agent_map: task_id → agent_id（从 assignments 获取正确映射）。
    """
    team = orch.create_team("team-1", "alpha", "lead-1")
    for i in range(1, n_tasks + 1):
        orch.register_agent(
            agent_id=f"agent-{i}",
            agent_name=f"worker-{i}",
            capabilities=frozenset({"code_generation"}),
        )
    specs = [
        TaskSpec(subject=f"task-{i}", description=f"desc-{i}", priority=i)
        for i in range(1, n_tasks + 1)
    ]
    run = await orch.start_from_plan(team, "objective", specs)
    task_ids = sorted(run.task_ids)
    agent_map = {a.task_id: a.agent_id for a in run.assignments if a.task_id}
    return run, task_ids, agent_map


# -- 测试 1：SimpleResultCollector.collect 反映 TaskStore 状态 --


@pytest.mark.asyncio
async def test_collector_reflects_taskstore_status(orchestrator: TeamOrchestrator):
    """SimpleResultCollector.collect 应从 TaskStore 聚合结果，而非读 run.results 内存副本。

    场景：两个 task 都通过 complete_task 标记为 completed，
    之后在 TaskStore 中将一个 task 改为 failed。
    collect 应反映 TaskStore 的 failed 状态。
    """
    run, task_ids, agent_map = await _setup_run(orchestrator, n_tasks=2)
    run_id = run.run_id

    # 通过 complete_task 完成两个 task（都成功）
    await orchestrator.complete_task(run_id, agent_map[task_ids[0]], task_ids[0], summary="ok-1")
    await orchestrator.complete_task(run_id, agent_map[task_ids[1]], task_ids[1], summary="ok-2")

    # 在 TaskStore 中将第一个 task 状态改为 failed
    task = orchestrator._task_store.load(task_ids[0])
    modified = dataclasses.replace(task, status=TaskStatus.FAILED.value, error="externally failed")
    orchestrator._task_store.save(modified)

    # SimpleResultCollector.collect 应反映 TaskStore 的 failed 状态
    collector = SimpleResultCollector(orchestrator)
    result = await collector.collect(run_id)

    assert result.failed_count == 1, (
        f"expected failed_count=1 from TaskStore, got {result.failed_count}; "
        f"results={[(r.task_id, r.status) for r in result.results]}"
    )
    assert result.completed_count == 1, (
        f"expected completed_count=1 from TaskStore, got {result.completed_count}"
    )


# -- 测试 2：_check_run_completion 基于 TaskStore 终态判定 FAILED --


@pytest.mark.asyncio
async def test_check_completion_uses_taskstore_failed(orchestrator: TeamOrchestrator):
    """_check_run_completion 应基于 TaskStore 终态判定 FAILED，而非读 run.results。

    场景：两个 task 都通过 complete_task 标记为 completed（run 进入 COMPLETED），
    之后在 TaskStore 中将一个 task 改为 failed。
    重置 run 为 RUNNING 后调用 _check_run_completion，应判定为 FAILED。
    """
    run, task_ids, agent_map = await _setup_run(orchestrator, n_tasks=2)
    run_id = run.run_id

    await orchestrator.complete_task(run_id, agent_map[task_ids[0]], task_ids[0], summary="ok-1")
    await orchestrator.complete_task(run_id, agent_map[task_ids[1]], task_ids[1], summary="ok-2")
    assert run.status == TeamRunStatus.COMPLETED

    # 在 TaskStore 中将第一个 task 状态改为 failed
    task = orchestrator._task_store.load(task_ids[0])
    modified = dataclasses.replace(task, status=TaskStatus.FAILED.value, error="externally failed")
    orchestrator._task_store.save(modified)

    # 重置 run 为 RUNNING 以便重新检查
    run.status = TeamRunStatus.RUNNING
    run.finished_at = None
    run.error = None

    await orchestrator._check_run_completion(run)

    assert run.status == TeamRunStatus.FAILED, (
        f"expected FAILED from TaskStore terminal state, got {run.status.value}"
    )


# -- 测试 3：_check_run_completion 忽略非终态 task（不引入回归）--


@pytest.mark.asyncio
async def test_check_completion_ignores_non_terminal(orchestrator: TeamOrchestrator):
    """_check_run_completion 不应将非终态 task（in_progress）误判为失败。

    场景：两个 task，只完成一个，另一个在 TaskStore 中仍是 in_progress。
    _check_run_completion 不应提前判定为 FAILED 或 COMPLETED。
    """
    run, task_ids, agent_map = await _setup_run(orchestrator, n_tasks=2)
    run_id = run.run_id

    # 只完成第一个 task
    await orchestrator.complete_task(run_id, agent_map[task_ids[0]], task_ids[0], summary="ok-1")

    # 第二个 task 在 TaskStore 中仍是 in_progress（start_from_plan 时被 claim→in_progress）
    task2 = orchestrator._task_store.load(task_ids[1])
    assert task2.status == TaskStatus.IN_PROGRESS.value, (
        f"expected second task to be in_progress, got {task2.status}"
    )

    # _check_run_completion 不应提前判定终态
    run.status = TeamRunStatus.RUNNING
    run.finished_at = None
    run.error = None
    await orchestrator._check_run_completion(run)

    assert run.status == TeamRunStatus.RUNNING, (
        f"expected RUNNING (not all tasks terminal), got {run.status.value}"
    )


# -- 测试 4：SimpleResultCollector.collect 与 collect_results 一致（含外部修改）--


@pytest.mark.asyncio
async def test_collector_matches_collect_results_after_external_change(
    orchestrator: TeamOrchestrator,
):
    """SimpleResultCollector.collect 和 collect_results 应在 TaskStore 外部修改后仍一致。

    两者都应通过 _aggregate_results 从 TaskStore 聚合。
    """
    run, task_ids, agent_map = await _setup_run(orchestrator, n_tasks=2)
    run_id = run.run_id

    await orchestrator.complete_task(run_id, agent_map[task_ids[0]], task_ids[0], summary="ok-1")
    await orchestrator.complete_task(run_id, agent_map[task_ids[1]], task_ids[1], error="boom")

    # 外部修改：将失败的 task 改为 completed
    task = orchestrator._task_store.load(task_ids[1])
    modified = dataclasses.replace(task, status=TaskStatus.COMPLETED.value, error=None)
    orchestrator._task_store.save(modified)

    collector = SimpleResultCollector(orchestrator)
    from_collect = await collector.collect(run_id)
    from_method = await orchestrator.collect_results(run_id)

    assert from_collect.completed_count == from_method.completed_count, (
        f"completed_count mismatch: collector={from_collect.completed_count}, "
        f"method={from_method.completed_count}"
    )
    assert from_collect.failed_count == from_method.failed_count, (
        f"failed_count mismatch: collector={from_collect.failed_count}, "
        f"method={from_method.failed_count}"
    )
    collect_status = {r.task_id: r.status for r in from_collect.results}
    method_status = {r.task_id: r.status for r in from_method.results}
    assert collect_status == method_status, (
        f"status mismatch: collector={collect_status}, method={method_status}"
    )
