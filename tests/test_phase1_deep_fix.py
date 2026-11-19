"""Phase 1 深度修复验证：超时类型区分 + TaskStore 接口完善。

P0-04: TaskWaitTimeoutError / TaskExecutionTimeoutError 作为 DomainError 子类，
       wait 超时抛 TaskWaitTimeoutError 而非 TaskTimeoutError。
P1-01: TaskStoreAdapter.complete_attempt 持久化 output/usage/cost_usd/artifacts；
       fail_attempt 区分 TIMED_OUT/CANCELLED 终态；
       expire_leases 真正回收过期租约；
       list_tasks 按 run_id 过滤；
       transition_task CAS 冲突抛异常；
       TaskAttempt 新增 artifacts 字段；
       persist_dir 启用后 Attempt 持久化到磁盘。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcc_core.runtime import (
    AttemptStatus,
    CASConflictError,
    InMemoryTaskStore,
    InvalidTransitionError,
    TaskExecutionTimeoutError,
    TaskStatus,
    TaskWaitTimeoutError,
    TokenUsage,
)
from mcc_core.runtime.errors import DomainError
from mcc_core.runtime.task_record import TaskAttempt
from mcc_harness.runtime import TaskStoreAdapter
from mcc_tools.task import TaskStore

# -- P0-04: 超时类型区分 --


class TestTimeoutErrorTypes:
    def test_task_wait_timeout_is_domain_error(self):
        err = TaskWaitTimeoutError("task-1", 5.0)
        assert isinstance(err, DomainError)
        assert err.code == "task_wait_timeout"
        assert err.retryable is True
        assert err.details["task_id"] == "task-1"
        assert err.details["timeout"] == 5.0

    def test_task_execution_timeout_is_domain_error(self):
        err = TaskExecutionTimeoutError("att-1", 10.0)
        assert isinstance(err, DomainError)
        assert err.code == "task_execution_timeout"
        assert err.retryable is True
        assert err.details["attempt_id"] == "att-1"
        assert err.details["timeout"] == 10.0

    def test_wait_timeout_distinct_from_execution_timeout(self):
        wait_err = TaskWaitTimeoutError("task-1", 5.0)
        exec_err = TaskExecutionTimeoutError("att-1", 5.0)
        assert type(wait_err) is not type(exec_err)
        assert wait_err.code != exec_err.code


# -- P1-01: TaskAttempt artifacts 字段 --


class TestTaskAttemptArtifacts:
    def test_attempt_has_artifacts_field(self):
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task-1",
            agent_id="a-1",
        )
        assert attempt.artifacts is None

    def test_attempt_artifacts_assignable(self):
        attempt = TaskAttempt(
            attempt_id="att-1",
            task_id="task-1",
            agent_id="a-1",
            artifacts={"report": "/path/to/report.md"},
        )
        assert attempt.artifacts == {"report": "/path/to/report.md"}


# -- P1-01: TaskStoreAdapter.complete_attempt 持久化完整结果 --


@pytest.fixture()
def adapter(tmp_path: Path) -> TaskStoreAdapter:
    store = TaskStore(tmp_path / "tasks")
    return TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")


class TestCompleteAttemptPersistsFullResult:
    async def test_complete_persists_output_usage_cost_artifacts(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        result = await adapter.complete_attempt(
            attempt.attempt_id,
            summary="done",
            output={"key": "value"},
            usage=usage,
            cost_usd=0.02,
            artifacts={"report": "/tmp/report.md"},
        )
        assert result.status is AttemptStatus.SUCCEEDED
        assert result.summary == "done"
        assert result.output == {"key": "value"}
        assert result.usage == usage
        assert result.cost_usd == 0.02
        assert result.artifacts == {"report": "/tmp/report.md"}

        reloaded = await adapter.get_attempt(attempt.attempt_id)
        assert reloaded is not None
        assert reloaded.output == {"key": "value"}
        assert reloaded.usage == usage
        assert reloaded.cost_usd == 0.02
        assert reloaded.artifacts == {"report": "/tmp/report.md"}


# -- P1-01: TaskStoreAdapter.fail_attempt 区分终态 --


class TestFailAttemptDistinguishesTerminalStatus:
    async def test_fail_default_is_failed(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        result = await adapter.fail_attempt(attempt.attempt_id, error="boom")
        assert result.status is AttemptStatus.FAILED

    async def test_fail_timed_out(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        result = await adapter.fail_attempt(
            attempt.attempt_id,
            error="timeout",
            status=AttemptStatus.TIMED_OUT,
        )
        assert result.status is AttemptStatus.TIMED_OUT
        assert result.is_terminal

    async def test_fail_cancelled(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        result = await adapter.fail_attempt(
            attempt.attempt_id,
            error="cancelled",
            status=AttemptStatus.CANCELLED,
        )
        assert result.status is AttemptStatus.CANCELLED
        assert result.is_terminal

    async def test_fail_rejects_non_terminal_status(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        with pytest.raises(InvalidTransitionError):
            await adapter.fail_attempt(
                attempt.attempt_id,
                error="x",
                status=AttemptStatus.RUNNING,
            )


# -- P1-01: TaskStoreAdapter.expire_leases 真正回收 --


class TestExpireLeases:
    async def test_expire_leases_reclaims_expired(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks")
        adapter = TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")
        task = await adapter.create_task("s", "d")
        claimed = await adapter.claim_task(task.task_id, "a-1", lease_seconds=0.01)
        assert claimed.status is TaskStatus.RUNNING
        await asyncio.sleep(0.05)
        expired = await adapter.expire_leases()
        assert task.task_id in expired
        refreshed = await adapter.get_task(task.task_id)
        assert refreshed.status is TaskStatus.PENDING

    async def test_expire_leases_skips_active(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        await adapter.claim_task(task.task_id, "a-1", lease_seconds=300)
        expired = await adapter.expire_leases()
        assert task.task_id not in expired


# -- P1-01: TaskStoreAdapter.list_tasks 按 run_id 过滤 --


class TestListTasksByRunId:
    async def test_list_filters_by_run_id(self, adapter: TaskStoreAdapter):
        t1 = await adapter.create_task("s1", "d1", run_id="run-A")
        t2 = await adapter.create_task("s2", "d2", run_id="run-B")
        t3 = await adapter.create_task("s3", "d3", run_id="run-A")

        run_a = await adapter.list_tasks(run_id="run-A")
        run_a_ids = {t.task_id for t in run_a}
        assert t1.task_id in run_a_ids
        assert t3.task_id in run_a_ids
        assert t2.task_id not in run_a_ids

    async def test_list_without_run_id_returns_all(self, adapter: TaskStoreAdapter):
        await adapter.create_task("s1", "d1", run_id="run-A")
        await adapter.create_task("s2", "d2", run_id="run-B")
        all_tasks = await adapter.list_tasks()
        assert len(all_tasks) >= 2


# -- P1-01: TaskStoreAdapter.transition_task CAS 冲突抛异常 --


class TestTransitionTaskCASConflict:
    async def test_cas_conflict_raises(self, adapter: TaskStoreAdapter):
        task = await adapter.create_task("s", "d")
        await adapter.transition_task(
            task.task_id,
            TaskStatus.PENDING,
            TaskStatus.DISPATCHED,
            agent_id="a-1",
        )
        with pytest.raises(CASConflictError):
            await adapter.transition_task(
                task.task_id,
                TaskStatus.PENDING,
                TaskStatus.ACCEPTED,
            )


# -- P1-01: persist_dir 持久化跨实例 --


class TestPersistDirSurvivesRestart:
    async def test_attempt_persisted_across_instances(self, tmp_path: Path):
        store_dir = tmp_path / "tasks"
        persist_dir = tmp_path / "attempts"

        store1 = TaskStore(store_dir)
        adapter1 = TaskStoreAdapter(store1, persist_dir=persist_dir)
        task = await adapter1.create_task("s", "d")
        attempt = await adapter1.create_attempt(task.task_id, "a-1", runtime_id="rt-1")
        await adapter1.complete_attempt(
            attempt.attempt_id,
            summary="done",
            output={"k": "v"},
            artifacts={"r": "/p"},
        )

        store2 = TaskStore(store_dir)
        adapter2 = TaskStoreAdapter(store2, persist_dir=persist_dir)
        reloaded = await adapter2.get_attempt(attempt.attempt_id)
        assert reloaded is not None
        assert reloaded.status is AttemptStatus.SUCCEEDED
        assert reloaded.summary == "done"
        assert reloaded.output == {"k": "v"}
        assert reloaded.artifacts == {"r": "/p"}


# -- P1-01: InMemoryTaskStore fail_attempt 区分终态 --


class TestInMemoryFailAttempt:
    async def test_fail_cancelled(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        result = await store.fail_attempt(
            attempt.attempt_id,
            error="cancelled",
            status=AttemptStatus.CANCELLED,
        )
        assert result.status is AttemptStatus.CANCELLED

    async def test_fail_timed_out(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        result = await store.fail_attempt(
            attempt.attempt_id,
            error="timeout",
            status=AttemptStatus.TIMED_OUT,
        )
        assert result.status is AttemptStatus.TIMED_OUT

    async def test_complete_persists_artifacts(self):
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        result = await store.complete_attempt(
            attempt.attempt_id,
            summary="done",
            artifacts={"r": "/p"},
        )
        assert result.artifacts == {"r": "/p"}
