"""PR-05 测试：Attempt 终态 CAS 原子更新，防止完成/取消互相覆盖。

P1-02 核心问题：
  Store 层 complete_attempt / fail_attempt 不检查当前状态、不验证 version、
  不阻止终态覆盖。竞态示例：
    T1 Dispatcher 读取 Attempt=RUNNING
    T2 Orchestrator cancel → fail_attempt(CANCELLED)
    T3 Runtime 完成事件到达
    T4 Dispatcher complete_attempt(SUCCEEDED)
  最终 CANCELLED 被覆盖为 SUCCEEDED。

修复后预期：
  - 终态不可被另一个终态覆盖（抛 CASConflictError）。
  - 同一终态重复写入幂等（返回已有状态）。
  - 迟到 RuntimeEvent 不能覆盖 Store 中已确定的状态。
"""

from __future__ import annotations

import pytest
from mcc_core.runtime.errors import CASConflictError
from mcc_core.runtime.task_record import AttemptStatus
from mcc_core.runtime.task_store import InMemoryTaskStore


@pytest.fixture
def store() -> InMemoryTaskStore:
    return InMemoryTaskStore()


async def _create_running_attempt(store: InMemoryTaskStore) -> str:
    """创建一个 RUNNING 状态的 attempt，返回 attempt_id。"""
    task = await store.create_task("s", "d")
    attempt = await store.create_attempt(task.task_id, "agent-1")
    await store.start_attempt(attempt.attempt_id)
    return attempt.attempt_id


class TestPR05AttemptCAS:
    """Attempt 终态 CAS 原子更新。"""

    async def test_complete_after_cancel_raises_cas_conflict(
        self,
        store: InMemoryTaskStore,
    ):
        """CANCELLED 后 complete_attempt 应抛 CASConflictError，不覆盖。"""
        attempt_id = await _create_running_attempt(store)

        # 先取消
        cancelled = await store.fail_attempt(
            attempt_id, error="cancelled", status=AttemptStatus.CANCELLED,
        )
        assert cancelled.status is AttemptStatus.CANCELLED

        # 再完成应抛 CASConflictError
        with pytest.raises(CASConflictError):
            await store.complete_attempt(attempt_id, summary="late success")

        # 状态仍为 CANCELLED
        attempt = await store.get_attempt(attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.CANCELLED

    async def test_cancel_after_complete_raises_cas_conflict(
        self,
        store: InMemoryTaskStore,
    ):
        """SUCCEEDED 后 fail_attempt(CANCELLED) 应抛 CASConflictError，不覆盖。"""
        attempt_id = await _create_running_attempt(store)

        # 先完成
        completed = await store.complete_attempt(attempt_id, summary="done")
        assert completed.status is AttemptStatus.SUCCEEDED

        # 再取消应抛 CASConflictError
        with pytest.raises(CASConflictError):
            await store.fail_attempt(
                attempt_id, error="late cancel", status=AttemptStatus.CANCELLED,
            )

        # 状态仍为 SUCCEEDED
        attempt = await store.get_attempt(attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.SUCCEEDED

    async def test_timeout_after_success_raises_cas_conflict(
        self,
        store: InMemoryTaskStore,
    ):
        """SUCCEEDED 后 fail_attempt(TIMED_OUT) 应抛 CASConflictError。"""
        attempt_id = await _create_running_attempt(store)

        await store.complete_attempt(attempt_id, summary="done")

        with pytest.raises(CASConflictError):
            await store.fail_attempt(
                attempt_id, error="late timeout", status=AttemptStatus.TIMED_OUT,
            )

        attempt = await store.get_attempt(attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.SUCCEEDED

    async def test_success_after_timeout_raises_cas_conflict(
        self,
        store: InMemoryTaskStore,
    ):
        """TIMED_OUT 后 complete_attempt 应抛 CASConflictError。"""
        attempt_id = await _create_running_attempt(store)

        await store.fail_attempt(
            attempt_id, error="timed out", status=AttemptStatus.TIMED_OUT,
        )

        with pytest.raises(CASConflictError):
            await store.complete_attempt(attempt_id, summary="late success")

        attempt = await store.get_attempt(attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.TIMED_OUT

    async def test_duplicate_complete_is_idempotent(
        self,
        store: InMemoryTaskStore,
    ):
        """同一终态重复写入幂等：第二次 complete_attempt 返回相同状态。"""
        attempt_id = await _create_running_attempt(store)

        first = await store.complete_attempt(attempt_id, summary="done")
        assert first.status is AttemptStatus.SUCCEEDED
        first_version = first.version

        # 第二次完成应幂等返回，不抛异常
        second = await store.complete_attempt(attempt_id, summary="done again")
        assert second.status is AttemptStatus.SUCCEEDED
        # 版本不应变化（幂等不 touch）
        assert second.version == first_version

    async def test_duplicate_cancel_is_idempotent(
        self,
        store: InMemoryTaskStore,
    ):
        """同一终态重复 fail_attempt 幂等。"""
        attempt_id = await _create_running_attempt(store)

        first = await store.fail_attempt(
            attempt_id, error="cancelled", status=AttemptStatus.CANCELLED,
        )
        assert first.status is AttemptStatus.CANCELLED
        first_version = first.version

        second = await store.fail_attempt(
            attempt_id, error="cancelled again", status=AttemptStatus.CANCELLED,
        )
        assert second.status is AttemptStatus.CANCELLED
        assert second.version == first_version

    async def test_duplicate_fail_is_idempotent(
        self,
        store: InMemoryTaskStore,
    ):
        """同一 FAILED 终态重复 fail_attempt 幂等。"""
        attempt_id = await _create_running_attempt(store)

        first = await store.fail_attempt(attempt_id, error="err")
        assert first.status is AttemptStatus.FAILED
        first_version = first.version

        second = await store.fail_attempt(attempt_id, error="err again")
        assert second.status is AttemptStatus.FAILED
        assert second.version == first_version

    async def test_failed_then_cancelled_raises_cas_conflict(
        self,
        store: InMemoryTaskStore,
    ):
        """FAILED 后 CANCELLED 应抛 CASConflictError（不同终态竞争）。"""
        attempt_id = await _create_running_attempt(store)

        await store.fail_attempt(attempt_id, error="failed")

        with pytest.raises(CASConflictError):
            await store.fail_attempt(
                attempt_id, error="cancel", status=AttemptStatus.CANCELLED,
            )

        attempt = await store.get_attempt(attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.FAILED
