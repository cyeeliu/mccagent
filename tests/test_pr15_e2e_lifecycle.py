"""PR-15: E2E 生命周期测试。

覆盖完整任务生命周期端到端场景：
- 多任务成功（PersistentRuntimeV2 并发多任务全部完成）
- 任务失败传播（失败结果通过事件传播）
- 取消传播（cancel 运行中任务，验证 TASK_CANCELLED）
- 执行超时（timeout_seconds 到期，验证 TASK_TIMED_OUT）
- 进程重启恢复（SqliteTaskStore 持久化，重启后任务可恢复）
- DAG 依赖失败（依赖任务失败，被依赖任务不执行）
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeEventType, TaskStatus
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class StubExecutor:
    """可配置桩执行器：支持成功/失败/延迟/取消。"""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        status: TaskStatus = TaskStatus.SUCCEEDED,
        summary: str = "done",
        error: str | None = None,
    ) -> None:
        self.delay = delay
        self.status = status
        self.summary = summary
        self.error = error
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.execute_count = 0

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.execute_count += 1
        self.started.set()
        try:
            if cancellation is not None and self.delay > 0:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(cancellation.wait()),
                        asyncio.create_task(asyncio.sleep(self.delay)),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
                if cancellation.is_cancelled:
                    return RuntimeResult(
                        task_id=task.task_id,
                        attempt_id=task.attempt_id,
                        runtime_id="stub",
                        agent_id=task.agent_id,
                        status=TaskStatus.CANCELLED,
                        summary="cancelled",
                    )
            else:
                await asyncio.sleep(self.delay)
            return RuntimeResult(
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                runtime_id="stub",
                agent_id=task.agent_id,
                status=self.status,
                summary=self.summary,
                error=self.error,
            )
        finally:
            self.finished.set()

    async def close(self) -> None:
        pass


def _make_task(*, timeout_seconds: float | None = None) -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="e2e-agent",
        objective="e2e test",
        timeout_seconds=timeout_seconds,
    )


async def _collect_events(
    runtime: Any,
    *targets: RuntimeEventType,
    timeout: float = 5.0,
) -> list[Any]:
    events: list[Any] = []
    seen: set[RuntimeEventType] = set()

    async def collector() -> None:
        async for ev in runtime.subscribe_events():
            events.append(ev)
            if ev.event_type in targets:
                seen.add(ev.event_type)
            if seen >= set(targets):
                break

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(collector(), timeout=timeout)
    return events


def _make_persistent_runtime(
    *,
    executor: StubExecutor | None = None,
    max_concurrency: int = 2,
) -> Any:
    from mcc_harness.runtime.persistent import PersistentRuntimeV2

    return PersistentRuntimeV2(
        executor or StubExecutor(),
        max_concurrency=max_concurrency,
        agent_id="e2e-persistent",
    )


class TestPR15E2EMultiTaskSuccess:
    """多任务成功：PersistentRuntimeV2 并发多任务全部完成。"""

    async def test_all_tasks_complete(self) -> None:
        rt = _make_persistent_runtime(
            executor=StubExecutor(delay=0.05),
            max_concurrency=3,
        )
        await rt.start()
        tasks = [_make_task() for _ in range(4)]
        submitted_ids = {t.task_id for t in tasks}

        events: list[Any] = []
        completed_ids: set[str] = set()

        async def collector() -> None:
            async for ev in rt.subscribe_events():
                events.append(ev)
                if ev.event_type is RuntimeEventType.TASK_COMPLETED:
                    completed_ids.add(ev.task_id)
                if completed_ids >= submitted_ids:
                    break

        collect_task = asyncio.create_task(
            asyncio.wait_for(collector(), timeout=10.0),
        )
        await asyncio.sleep(0.05)

        for task in tasks:
            await rt.submit(task)

        with contextlib.suppress(TimeoutError):
            await collect_task
        await rt.stop()

        assert completed_ids == submitted_ids, (
            f"未所有任务完成: completed={completed_ids}, "
            f"submitted={submitted_ids}"
        )


class TestPR15E2EFailurePropagation:
    """任务失败传播：失败结果通过事件传播。"""

    async def test_failed_event_emitted(self) -> None:
        rt = _make_persistent_runtime(
            executor=StubExecutor(
                delay=0.02,
                status=TaskStatus.FAILED,
                error="boom",
            ),
        )
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_FAILED, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task
        await rt.stop()

        failed_events = [
            ev for ev in events
            if ev.event_type is RuntimeEventType.TASK_FAILED
        ]
        assert len(failed_events) >= 1, "未收到 TASK_FAILED 事件"
        assert failed_events[0].task_id == task.task_id


class TestPR15E2ECancelPropagation:
    """取消传播：cancel 运行中任务，验证 TASK_CANCELLED。"""

    async def test_cancel_running_task(self) -> None:
        executor = StubExecutor(delay=10.0)
        rt = _make_persistent_runtime(executor=executor)
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_CANCELLED, timeout=5.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        await executor.started.wait()
        await rt.cancel_task(task.task_id, task.attempt_id, "e2e cancel")
        events = await collect_task
        await rt.stop()

        cancelled_events = [
            ev for ev in events
            if ev.event_type is RuntimeEventType.TASK_CANCELLED
        ]
        assert len(cancelled_events) >= 1, "未收到 TASK_CANCELLED 事件"
        assert cancelled_events[0].task_id == task.task_id


class TestPR15E2EExecutionTimeout:
    """执行超时：timeout_seconds 到期，验证 TASK_TIMED_OUT。"""

    async def test_timeout_emits_timed_out(self) -> None:
        rt = _make_persistent_runtime(executor=StubExecutor(delay=5.0))
        await rt.start()
        task = _make_task(timeout_seconds=0.2)

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task
        await rt.stop()

        timed_out_events = [
            ev for ev in events
            if ev.event_type is RuntimeEventType.TASK_TIMED_OUT
        ]
        assert len(timed_out_events) >= 1, "未收到 TASK_TIMED_OUT 事件"
        assert timed_out_events[0].task_id == task.task_id


class TestPR15E2ERestartRecovery:
    """进程重启恢复：SqliteTaskStore 持久化，重启后任务可恢复。"""

    async def test_task_survives_restart(self, tmp_path: Path) -> None:
        from mcc_core.runtime.enums import TaskStatus
        from mcc_core.runtime.task_store_sqlite import SqliteTaskStore

        db_path = tmp_path / "tasks.db"

        store1 = SqliteTaskStore(db_path)
        await store1.init()
        record = await store1.create_task(
            "restart-test",
            "test description",
            max_attempts=2,
        )
        attempt = await store1.create_attempt(
            record.task_id,
            "agent-1",
            runtime_id="rt-1",
        )
        await store1.start_attempt(attempt.attempt_id)
        await store1.complete_attempt(
            attempt.attempt_id,
            summary="completed before restart",
        )
        await store1.close()

        store2 = SqliteTaskStore(db_path)
        await store2.init()

        recovered = await store2.get_task(record.task_id)
        assert recovered is not None, "重启后任务丢失"
        assert recovered.subject == "restart-test"
        assert recovered.status is TaskStatus.SUCCEEDED, (
            f"重启后任务状态错误: {recovered.status}"
        )

        attempts = await store2.list_attempts(record.task_id)
        assert len(attempts) == 1, "重启后 attempt 丢失"
        from mcc_core.runtime.task_record import AttemptStatus

        assert attempts[0].status is AttemptStatus.SUCCEEDED, (
            f"重启后 attempt 状态错误: {attempts[0].status}"
        )
        await store2.close()

    async def test_inflight_task_recovered(self, tmp_path: Path) -> None:
        from mcc_core.runtime.enums import TaskStatus
        from mcc_core.runtime.task_record import AttemptStatus
        from mcc_core.runtime.task_store_sqlite import SqliteTaskStore

        db_path = tmp_path / "tasks.db"

        store1 = SqliteTaskStore(db_path)
        await store1.init()
        record = await store1.create_task(
            "inflight-test",
            "inflight task",
            max_attempts=2,
        )
        attempt = await store1.create_attempt(
            record.task_id,
            "agent-1",
            runtime_id="rt-1",
        )
        await store1.start_attempt(attempt.attempt_id)
        await store1.close()

        store2 = SqliteTaskStore(db_path)
        await store2.init()

        recovered = await store2.get_task(record.task_id)
        assert recovered is not None
        assert recovered.status is TaskStatus.PENDING, (
            f"重启后任务状态错误: {recovered.status}"
        )

        attempts = await store2.list_attempts(record.task_id)
        assert len(attempts) == 1
        assert attempts[0].status is AttemptStatus.RUNNING, (
            f"重启后 attempt 状态错误: {attempts[0].status}"
        )
        await store2.close()


class TestPR15E2EDagDependencyFailure:
    """DAG 依赖失败：依赖任务失败，被依赖任务不执行。"""

    async def test_dependent_not_executed_on_failure(self, tmp_path: Path) -> None:
        from mcc_harness import (
            AgentRegistry,
            MessageBus,
            PlanStore,
            RuntimeCoordinator,
            SimplePlanner,
            TaskDispatcher,
            TaskSpec,
            TeamOrchestrator,
        )
        from mcc_harness.runtime import TaskStoreAdapter
        from mcc_tools import TaskStore

        registry = AgentRegistry(heartbeat_timeout=30.0)
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        task_store = TaskStore(tmp_path / ".tasks")
        coordinator = RuntimeCoordinator()
        adapter = TaskStoreAdapter(task_store)
        dispatcher = TaskDispatcher(adapter, coordinator)
        orch = TeamOrchestrator(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            coordinator=coordinator,
            dispatcher=dispatcher,
        )

        # 注册 2 个 agent：CapabilityScheduler 每个 agent 只分配一个 task，
        # 因此 base → agent-1，dependent → agent-2。
        orch.register_agent(
            agent_id="agent-1",
            agent_name="worker-1",
            capabilities=frozenset({"code_generation"}),
        )
        orch.register_agent(
            agent_id="agent-2",
            agent_name="worker-2",
            capabilities=frozenset({"code_generation"}),
        )
        team = orch.create_team("team-1", "alpha", "lead-1")

        executed_subjects: list[str] = []

        class FailExecutor(StubExecutor):
            """base task 执行器：始终返回 FAILED。"""

            async def execute(
                self,
                task: RuntimeTask,
                *,
                emit: Any | None = None,
                cancellation: Any | None = None,
            ) -> RuntimeResult:
                # 等待延迟，确保 wait_for_result 的 listener 已注册，
                # 避免事件流竞态导致事件丢失。
                await asyncio.sleep(self.delay)
                executed_subjects.append(task.objective)
                return RuntimeResult(
                    task_id=task.task_id,
                    attempt_id=task.attempt_id,
                    runtime_id="stub-fail",
                    agent_id=task.agent_id,
                    status=TaskStatus.FAILED,
                    error="base failed",
                )

        class TrackExecutor(StubExecutor):
            """dependent task 执行器：记录执行但不应被调用。"""

            async def execute(
                self,
                task: RuntimeTask,
                *,
                emit: Any | None = None,
                cancellation: Any | None = None,
            ) -> RuntimeResult:
                executed_subjects.append(task.objective)
                return await super().execute(
                    task, emit=emit, cancellation=cancellation,
                )

        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        rt1 = PersistentRuntimeV2(
            FailExecutor(delay=0.02),
            max_concurrency=1,
            agent_id="dag-agent-1",
        )
        rt2 = PersistentRuntimeV2(
            TrackExecutor(delay=0.02),
            max_concurrency=1,
            agent_id="dag-agent-2",
        )
        await coordinator.register_and_start(rt1)
        await coordinator.register_and_start(rt2)
        orch.bind_runtime("agent-1", rt1.runtime_id)
        orch.bind_runtime("agent-2", rt2.runtime_id)

        class DagPlanner(SimplePlanner):
            async def plan(self, objective, *, context=None):
                return [
                    TaskSpec(subject="base", description="base task"),
                    TaskSpec(
                        subject="dependent",
                        description="dep task",
                        blocked_by=("base",),
                    ),
                ]

        orch._planner = DagPlanner()
        run = await orch.start(team, "base\ndependent")
        run = await orch.execute_run(run.run_id)

        from mcc_harness import TeamRunStatus

        assert run.status is TeamRunStatus.FAILED, (
            f"预期 FAILED，实际 {run.status}"
        )
        assert "base" in executed_subjects, "base 任务未执行"
        assert "dependent" not in executed_subjects, (
            "依赖任务在 base 失败后不应执行"
        )
        await coordinator.shutdown_all()
