"""PR-06 测试：TaskStoreAdapter 状态语义与 InMemoryTaskStore 对齐。

P1-01 核心问题：
  1. _ENUM_TO_STATUS 将 DISPATCHED、ACCEPTED、RUNNING 全部映射成 "claimed"；
     读取时 "claimed" 只能映射回 DISPATCHED。Task 状态无法区分。
  2. complete_attempt / fail_attempt 只更新 Attempt，不更新底层 TaskRecord。
  3. create_assignment 不设置 lease_expires_at。
  4. complete_attempt / fail_attempt 无 CAS 终态保护。

修复后预期：
  - RUNNING 映射到 "in_progress"，不折叠为 "claimed"。
  - complete_attempt 同时更新 TaskRecord 为 SUCCEEDED。
  - fail_attempt 同时更新 TaskRecord 为 FAILED（当不可重试时）。
  - create_assignment 设置 lease_expires_at。
  - adapter 的 complete_attempt / fail_attempt 有 CAS 终态保护。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.errors import CASConflictError
from mcc_core.runtime.task_record import AttemptStatus
from mcc_harness.runtime.task_store_adapter import TaskStoreAdapter


def _make_adapter(tmp_path: Path) -> TaskStoreAdapter:
    """创建一个使用 mcc_tools TaskStore 的 TaskStoreAdapter。"""
    from mcc_tools import TaskStore

    store = TaskStore(tmp_path / ".tasks")
    return TaskStoreAdapter(store, persist_dir=tmp_path / ".persist")


@pytest.fixture
def adapter(tmp_path: Path) -> TaskStoreAdapter:
    return _make_adapter(tmp_path)


async def _create_running_attempt(adapter: TaskStoreAdapter) -> tuple[str, str]:
    """创建一个 RUNNING 状态的 attempt，返回 (task_id, attempt_id)。"""
    record = await adapter.create_task("s", "d")
    attempt = await adapter.create_attempt(record.task_id, "agent-1")
    await adapter.start_attempt(attempt.attempt_id)
    return record.task_id, attempt.attempt_id


class TestPR06AdapterStatusMapping:
    """状态映射不折叠：RUNNING 不映射为 "claimed"。"""

    async def test_running_maps_to_running(
        self,
        adapter: TaskStoreAdapter,
    ):
        """RUNNING 状态映射到 "running"，不折叠为 "claimed" 或 "in_progress"。

        P1-01 修复后状态映射不折叠：每个 enum 值映射为唯一字符串，
        DISPATCHED → "dispatched"，ACCEPTED → "accepted"，RUNNING → "running"。
        """
        from mcc_harness.runtime.task_store_adapter import _ENUM_TO_STATUS

        assert _ENUM_TO_STATUS[TaskStatus.RUNNING] == "running"
        assert _ENUM_TO_STATUS[TaskStatus.DISPATCHED] == "dispatched"
        assert _ENUM_TO_STATUS[TaskStatus.ACCEPTED] == "accepted"

    async def test_transition_to_running_preserves_status(
        self,
        adapter: TaskStoreAdapter,
    ):
        """transition_task 到 RUNNING 后，get_task 返回 RUNNING 而非 DISPATCHED。"""
        record = await adapter.create_task("s", "d")
        await adapter.transition_task(
            record.task_id,
            TaskStatus.PENDING,
            TaskStatus.DISPATCHED,
            agent_id="a-1",
        )
        await adapter.transition_task(
            record.task_id,
            TaskStatus.DISPATCHED,
            TaskStatus.RUNNING,
            agent_id="a-1",
        )
        task = await adapter.get_task(record.task_id)
        assert task is not None
        assert task.status is TaskStatus.RUNNING


class TestPR06AdapterCompleteUpdatesTaskRecord:
    """complete_attempt 同时更新底层 TaskRecord。"""

    async def test_complete_attempt_sets_task_succeeded(
        self,
        adapter: TaskStoreAdapter,
    ):
        """complete_attempt 后 TaskRecord 状态为 SUCCEEDED。"""
        task_id, attempt_id = await _create_running_attempt(adapter)

        await adapter.complete_attempt(attempt_id, summary="done")

        task = await adapter.get_task(task_id)
        assert task is not None
        assert task.status is TaskStatus.SUCCEEDED

    async def test_fail_attempt_sets_task_failed_when_no_retry(
        self,
        adapter: TaskStoreAdapter,
    ):
        """fail_attempt 后 TaskRecord 状态为 FAILED（当不可重试时）。"""
        record = await adapter.create_task("s", "d", max_attempts=1)
        attempt = await adapter.create_attempt(record.task_id, "agent-1")
        await adapter.start_attempt(attempt.attempt_id)

        await adapter.fail_attempt(attempt.attempt_id, error="boom")

        task = await adapter.get_task(record.task_id)
        assert task is not None
        assert task.status is TaskStatus.FAILED


class TestPR06AdapterAssignmentLease:
    """create_assignment 设置 lease_expires_at。"""

    async def test_create_assignment_sets_lease_expires_at(
        self,
        adapter: TaskStoreAdapter,
    ):
        """create_assignment 根据 lease_seconds 设置 lease_expires_at。"""
        record = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(record.task_id, "agent-1")

        assignment = await adapter.create_assignment(
            record.task_id,
            attempt.attempt_id,
            "agent-1",
            lease_seconds=120,
        )

        assert assignment.lease_expires_at is not None
        # lease_expires_at 应在 assigned_at 之后约 120 秒
        delta = assignment.lease_expires_at - assignment.assigned_at
        assert timedelta(seconds=110) < delta < timedelta(seconds=130)

    async def test_create_assignment_default_lease(
        self,
        adapter: TaskStoreAdapter,
    ):
        """create_assignment 默认 lease_seconds=300。"""
        record = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(record.task_id, "agent-1")

        assignment = await adapter.create_assignment(
            record.task_id,
            attempt.attempt_id,
            "agent-1",
        )

        assert assignment.lease_expires_at is not None
        delta = assignment.lease_expires_at - assignment.assigned_at
        assert timedelta(seconds=290) < delta < timedelta(seconds=310)


class TestPR06AdapterCAS:
    """Adapter 的 complete_attempt / fail_attempt 有 CAS 终态保护。"""

    async def test_complete_after_cancel_raises_cas(
        self,
        adapter: TaskStoreAdapter,
    ):
        """CANCELLED 后 complete_attempt 抛 CASConflictError。"""
        _, attempt_id = await _create_running_attempt(adapter)

        await adapter.fail_attempt(
            attempt_id, error="cancelled", status=AttemptStatus.CANCELLED,
        )

        with pytest.raises(CASConflictError):
            await adapter.complete_attempt(attempt_id, summary="late")

    async def test_cancel_after_complete_raises_cas(
        self,
        adapter: TaskStoreAdapter,
    ):
        """SUCCEEDED 后 fail_attempt(CANCELLED) 抛 CASConflictError。"""
        _, attempt_id = await _create_running_attempt(adapter)

        await adapter.complete_attempt(attempt_id, summary="done")

        with pytest.raises(CASConflictError):
            await adapter.fail_attempt(
                attempt_id, error="late cancel", status=AttemptStatus.CANCELLED,
            )

    async def test_duplicate_complete_is_idempotent(
        self,
        adapter: TaskStoreAdapter,
    ):
        """同一终态重复写入幂等。"""
        _, attempt_id = await _create_running_attempt(adapter)

        first = await adapter.complete_attempt(attempt_id, summary="done")
        first_version = first.version

        second = await adapter.complete_attempt(attempt_id, summary="done again")
        assert second.status is AttemptStatus.SUCCEEDED
        assert second.version == first_version

    async def test_duplicate_cancel_is_idempotent(
        self,
        adapter: TaskStoreAdapter,
    ):
        """同一 CANCELLED 终态重复 fail_attempt 幂等。"""
        _, attempt_id = await _create_running_attempt(adapter)

        first = await adapter.fail_attempt(
            attempt_id, error="cancelled", status=AttemptStatus.CANCELLED,
        )
        first_version = first.version

        second = await adapter.fail_attempt(
            attempt_id, error="cancelled again", status=AttemptStatus.CANCELLED,
        )
        assert second.status is AttemptStatus.CANCELLED
        assert second.version == first_version
