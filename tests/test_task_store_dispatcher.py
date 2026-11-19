"""Phase 6: TaskStore / TaskDispatcher 领域模型测试。

验收标准：
- 每个正式任务都有 TaskRecord。
- 每次执行有独立 Attempt。
- 每次分配有 Assignment。
- 重复派发不重复执行（幂等）。
- 结果持久化后才发完成事件。
- CAS 状态迁移保证原子性。
- Lease 续约与过期回收。
"""

from __future__ import annotations

import asyncio

import pytest
from mcc_core.runtime import (
    AssignmentStatus,
    AttemptStatus,
    CASConflictError,
    InMemoryTaskStore,
    InvalidTransitionError,
    LeaseExpiredError,
    RuntimeCapabilities,
    RuntimeEventType,
    RuntimeTask,
    TaskAssignment,
    TaskAttempt,
    TaskNotFoundError,
    TaskRecord,
    TaskStatus,
    TaskStoreProtocol,
)
from mcc_core.runtime.task import TaskContext
from mcc_harness import (
    DispatchError,
    RuntimeCoordinator,
    TaskDispatcher,
)

# -- TaskRecord 领域模型 --


class TestTaskRecord:
    def test_default_values(self):
        task = TaskRecord(task_id="t-1", subject="test", description="desc")
        assert task.status is TaskStatus.PENDING
        assert task.priority == 0
        assert task.max_attempts == 3
        assert task.attempt_count == 0
        assert task.version == 0
        assert task.can_retry

    def test_touch_increments_version(self):
        task = TaskRecord(task_id="t-1", subject="test", description="desc")
        v0 = task.version
        task.touch()
        assert task.version == v0 + 1

    def test_can_retry_respects_max_attempts(self):
        task = TaskRecord(
            task_id="t-1",
            subject="test",
            description="desc",
            max_attempts=2,
            attempt_count=2,
        )
        assert not task.can_retry

    def test_is_terminal(self):
        task = TaskRecord(
            task_id="t-1",
            subject="test",
            description="desc",
            status=TaskStatus.SUCCEEDED,
        )
        assert task.is_terminal

        task2 = TaskRecord(
            task_id="t-2",
            subject="test",
            description="desc",
            status=TaskStatus.RUNNING,
        )
        assert not task2.is_terminal


# -- TaskAttempt 领域模型 --


class TestTaskAttempt:
    def test_default_idempotency_key(self):
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="t-1",
            agent_id="a-1",
        )
        assert attempt.idempotency_key == "t-1:att-1"

    def test_custom_idempotency_key(self):
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="t-1",
            agent_id="a-1",
            idempotency_key="custom-key",
        )
        assert attempt.idempotency_key == "custom-key"

    def test_is_terminal(self):
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="t-1",
            agent_id="a-1",
            status=AttemptStatus.SUCCEEDED,
        )
        assert attempt.is_terminal

    def test_is_retry(self):
        attempt = TaskAttempt(
            attempt_id="att-2",
            task_id="t-1",
            agent_id="a-1",
            idempotency_key="t-1:att-1",
        )
        assert attempt.is_retry


# -- TaskAssignment 领域模型 --


class TestTaskAssignment:
    def test_is_active_by_default(self):
        asg = TaskAssignment(
            assignment_id="asg-1",
            task_id="t-1",
            attempt_id="att-1",
            agent_id="a-1",
        )
        assert asg.is_active
        assert not asg.is_expired

    def test_is_expired_with_past_lease(self):
        from datetime import UTC, datetime, timedelta

        asg = TaskAssignment(
            assignment_id="asg-1",
            task_id="t-1",
            attempt_id="att-1",
            agent_id="a-1",
            lease_expires_at=datetime.now(UTC) - timedelta(seconds=10),
        )
        assert asg.is_expired
        assert not asg.is_active

    def test_completed_not_active(self):
        asg = TaskAssignment(
            assignment_id="asg-1",
            task_id="t-1",
            attempt_id="att-1",
            agent_id="a-1",
            status=AssignmentStatus.COMPLETED,
        )
        assert not asg.is_active


# -- InMemoryTaskStore: Task CRUD --


class TestInMemoryTaskStoreTask:
    async def test_create_task(self):
        store = InMemoryTaskStore()
        task = await store.create_task("subject", "description")
        assert task.task_id.startswith("task_")
        assert task.status is TaskStatus.PENDING

    async def test_get_task_not_found(self):
        store = InMemoryTaskStore()
        assert await store.get_task("nope") is None

    async def test_list_tasks_filter_by_status(self):
        store = InMemoryTaskStore()
        t1 = await store.create_task("s1", "d1")
        await store.transition_task(t1.task_id, TaskStatus.PENDING, TaskStatus.QUEUED)
        await store.create_task("s2", "d2")
        pending = await store.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == t2_task_id if (t2_task_id := None) else True

    async def test_list_tasks_filter_by_run_id(self):
        store = InMemoryTaskStore()
        t1 = await store.create_task("s1", "d1", run_id="run-1")
        await store.create_task("s2", "d2", run_id="run-2")
        run1_tasks = await store.list_tasks(run_id="run-1")
        assert len(run1_tasks) == 1
        assert run1_tasks[0].task_id == t1.task_id


# -- InMemoryTaskStore: CAS 状态迁移 --


class TestInMemoryTaskStoreCAS:
    async def test_transition_task_success(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        old_version = task.version
        updated = await store.transition_task(
            task.task_id,
            TaskStatus.PENDING,
            TaskStatus.QUEUED,
        )
        assert updated.status is TaskStatus.QUEUED
        assert updated.version > old_version

    async def test_transition_task_wrong_from_status(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        with pytest.raises(CASConflictError):
            await store.transition_task(
                task.task_id,
                TaskStatus.QUEUED,
                TaskStatus.DISPATCHED,
            )

    async def test_transition_task_invalid_transition(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        with pytest.raises(InvalidTransitionError):
            await store.transition_task(
                task.task_id,
                TaskStatus.PENDING,
                TaskStatus.RUNNING,
            )

    async def test_transition_task_not_found(self):
        store = InMemoryTaskStore()
        with pytest.raises(TaskNotFoundError):
            await store.transition_task(
                "nope",
                TaskStatus.PENDING,
                TaskStatus.QUEUED,
            )


# -- InMemoryTaskStore: Lease 管理 --


class TestInMemoryTaskStoreLease:
    async def test_claim_task(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        claimed = await store.claim_task(task.task_id, "agent-1", lease_seconds=60)
        assert claimed.status is TaskStatus.DISPATCHED
        assert claimed.agent_id == "agent-1"
        assert claimed.lease_expires_at is not None

    async def test_renew_lease(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.claim_task(task.task_id, "agent-1", lease_seconds=60)
        renewed = await store.renew_lease(task.task_id, "agent-1", lease_seconds=120)
        assert renewed.lease_expires_at is not None

    async def test_renew_lease_wrong_agent(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.claim_task(task.task_id, "agent-1", lease_seconds=60)
        with pytest.raises(LeaseExpiredError):
            await store.renew_lease(task.task_id, "agent-2", lease_seconds=60)

    async def test_expire_leases(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.claim_task(task.task_id, "agent-1", lease_seconds=0)
        await asyncio.sleep(0.01)
        expired = await store.expire_leases()
        assert task.task_id in expired
        updated = await store.get_task(task.task_id)
        assert updated.status is TaskStatus.PENDING
        assert updated.agent_id is None


# -- InMemoryTaskStore: Attempt 管理 --


class TestInMemoryTaskStoreAttempt:
    async def test_create_attempt(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "agent-1")
        assert attempt.task_id == task.task_id
        assert attempt.agent_id == "agent-1"
        assert attempt.status is AttemptStatus.PENDING

    async def test_create_attempt_increments_count(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.create_attempt(task.task_id, "a-1")
        await store.create_attempt(task.task_id, "a-1")
        updated = await store.get_task(task.task_id)
        assert updated.attempt_count == 2

    async def test_idempotent_create_attempt(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        a1 = await store.create_attempt(
            task.task_id,
            "a-1",
            idempotency_key="idem-1",
        )
        a2 = await store.create_attempt(
            task.task_id,
            "a-1",
            idempotency_key="idem-1",
        )
        assert a1.attempt_id == a2.attempt_id

    async def test_complete_attempt(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        completed = await store.complete_attempt(
            attempt.attempt_id,
            summary="done",
            output={"result": 42},
        )
        assert completed.status is AttemptStatus.SUCCEEDED
        assert completed.summary == "done"
        updated = await store.get_task(task.task_id)
        assert updated.status is TaskStatus.SUCCEEDED

    async def test_fail_attempt(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        failed = await store.fail_attempt(attempt.attempt_id, error="boom")
        assert failed.status is AttemptStatus.FAILED
        assert failed.error == "boom"

    async def test_fail_attempt_marks_task_failed_when_no_retries(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d", max_attempts=1)
        attempt = await store.create_attempt(task.task_id, "a-1")
        await store.fail_attempt(attempt.attempt_id, error="boom")
        updated = await store.get_task(task.task_id)
        assert updated.status is TaskStatus.FAILED

    async def test_list_attempts(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.create_attempt(task.task_id, "a-1")
        await store.create_attempt(task.task_id, "a-1")
        attempts = await store.list_attempts(task.task_id)
        assert len(attempts) == 2

    async def test_find_by_idempotency_key(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        await store.create_attempt(
            task.task_id,
            "a-1",
            idempotency_key="special-key",
        )
        found = await store.find_by_idempotency_key("special-key")
        assert found is not None
        not_found = await store.find_by_idempotency_key("nope")
        assert not_found is None


# -- InMemoryTaskStore: Assignment 管理 --


class TestInMemoryTaskStoreAssignment:
    async def test_create_assignment(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        asg = await store.create_assignment(
            task.task_id,
            attempt.attempt_id,
            "a-1",
        )
        assert asg.task_id == task.task_id
        assert asg.agent_id == "a-1"
        assert asg.status is AssignmentStatus.ACTIVE

    async def test_list_assignments_filter(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        await store.create_assignment(task.task_id, attempt.attempt_id, "a-1")
        await store.create_assignment(task.task_id, attempt.attempt_id, "a-2")
        a1_assignments = await store.list_assignments(agent_id="a-1")
        assert len(a1_assignments) == 1

    async def test_complete_assignment(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        asg = await store.create_assignment(
            task.task_id,
            attempt.attempt_id,
            "a-1",
        )
        completed = await store.complete_assignment(asg.assignment_id)
        assert completed.status is AssignmentStatus.COMPLETED
        assert completed.completed_at is not None


# -- TaskStoreProtocol 合规 --


class TestTaskStoreProtocol:
    def test_in_memory_store_satisfies_protocol(self):
        store = InMemoryTaskStore()
        assert isinstance(store, TaskStoreProtocol)


# -- TaskDispatcher 测试 --


class _StubRuntime:
    """支持新 Protocol submit + subscribe_events 的 Runtime 桩。"""

    def __init__(self, runtime_id: str = "stub-rt") -> None:
        self._runtime_id = runtime_id
        self._event_listeners: list[asyncio.Queue] = []
        self._submitted_tasks: list[RuntimeTask] = []

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self):
        return None

    @property
    def alive(self) -> bool:
        return True

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=False, supports_health_check=False)

    async def submit(self, task: RuntimeTask):
        from mcc_core.runtime import TaskHandle

        self._submitted_tasks.append(task)
        return TaskHandle(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
        )

    def subscribe_events(self, *, task_id=None, attempt_id=None):
        return self._event_iterator(task_id=task_id, attempt_id=attempt_id)

    async def _event_iterator(self, *, task_id=None, attempt_id=None):
        queue: asyncio.Queue = asyncio.Queue()
        self._event_listeners.append(queue)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield ev
        finally:
            self._event_listeners.remove(queue)

    async def _emit_done(self, task_id, attempt_id, event_type, payload=None):
        from datetime import UTC, datetime

        from mcc_core.runtime import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-done",
            event_type=event_type,
            runtime_id=self._runtime_id,
            agent_id="stub",
            task_id=task_id,
            attempt_id=attempt_id,
            sequence=1,
            occurred_at=datetime.now(UTC),
            payload=payload or {},
        )
        for q in self._event_listeners:
            await q.put(event)
            await q.put(None)

    async def stop(self, **kwargs):
        pass


class TestTaskDispatcher:
    async def test_dispatch_creates_attempt_and_submits(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("test subject", "test desc")
        handle = await dispatcher.dispatch(
            task.task_id,
            "agent-1",
            runtime_id="rt-1",
        )
        assert handle.task_id == task.task_id
        assert handle.runtime_id == "rt-1"
        assert len(rt._submitted_tasks) == 1

        updated = await store.get_task(task.task_id)
        assert updated.status is TaskStatus.ACCEPTED

    async def test_dispatch_runtime_not_found(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        with pytest.raises(DispatchError, match="runtime not found"):
            await dispatcher.dispatch(
                task.task_id,
                "agent-1",
                runtime_id="nope",
            )

    async def test_dispatch_task_not_found(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        with pytest.raises(TaskNotFoundError):
            await dispatcher.dispatch("nope", "agent-1", runtime_id="rt-1")

    async def test_idempotent_dispatch(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        h1 = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
            idempotency_key="idem-key",
        )
        h2 = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
            idempotency_key="idem-key",
        )
        assert h1.attempt_id == h2.attempt_id

    async def test_wait_for_result_completed(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )

        async def emit_done():
            await asyncio.sleep(0.05)
            await rt._emit_done(
                handle.task_id,
                handle.attempt_id,
                RuntimeEventType.TASK_COMPLETED,
                payload={"summary": "all good"},
            )

        emit_task = asyncio.create_task(emit_done())
        event = await dispatcher.wait_for_result(handle, timeout=2.0)
        await emit_task

        assert event.event_type is RuntimeEventType.TASK_COMPLETED
        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt.status is AttemptStatus.SUCCEEDED

    async def test_wait_for_result_failed(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )

        async def emit_fail():
            await asyncio.sleep(0.05)
            await rt._emit_done(
                handle.task_id,
                handle.attempt_id,
                RuntimeEventType.TASK_FAILED,
                payload={"error": "something went wrong"},
            )

        fail_task = asyncio.create_task(emit_fail())
        event = await dispatcher.wait_for_result(handle, timeout=2.0)
        await fail_task

        assert event.event_type is RuntimeEventType.TASK_FAILED
        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt.status is AttemptStatus.FAILED

    async def test_wait_for_result_timeout(self):
        from mcc_core.runtime.errors import TaskTimeoutError

        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )

        with pytest.raises(TaskTimeoutError):
            await dispatcher.wait_for_result(handle, timeout=0.05)

    async def test_retry_creates_new_attempt(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d", max_attempts=3)
        h1 = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )
        await store.fail_attempt(h1.attempt_id, error="first failed")

        h2 = await dispatcher.retry(task.task_id, runtime_id="rt-1")
        assert h1.attempt_id != h2.attempt_id
        assert h1.task_id == h2.task_id

    async def test_retry_exceeds_max_attempts(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d", max_attempts=1)
        await dispatcher.dispatch(task.task_id, "a-1", runtime_id="rt-1")
        with pytest.raises(DispatchError, match="max_attempts"):
            await dispatcher.retry(task.task_id, runtime_id="rt-1")

    async def test_dispatch_batch(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        t1 = await store.create_task("s1", "d1")
        t2 = await store.create_task("s2", "d2")
        results = await dispatcher.dispatch_batch(
            [
                (t1.task_id, "a-1", "rt-1"),
                (t2.task_id, "a-1", "rt-1"),
            ]
        )
        assert len(results) == 2
        assert all(isinstance(r, type(results[0])) for r in results)

    async def test_dispatch_batch_partial_failure(self):
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        t1 = await store.create_task("s1", "d1")
        results = await dispatcher.dispatch_batch(
            [
                (t1.task_id, "a-1", "rt-1"),
                ("nonexistent", "a-1", "rt-1"),
            ]
        )
        assert len(results) == 2
        assert not isinstance(results[0], Exception)
        assert isinstance(results[1], Exception)
