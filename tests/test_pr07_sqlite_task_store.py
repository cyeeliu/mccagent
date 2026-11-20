"""PR-07 测试：SQLite TaskStore 可靠持久化。

P1-03 + P1-11 核心问题：
  1. JSON 持久化无跨进程事务语义，并发写入可能丢失/重复 Attempt。
  2. JSON 损坏时静默当作空数据启动，导致重复执行。
  3. 同步文件写入阻塞事件循环。

修复后预期：
  - SQLite 事务保证原子性。
  - idempotency_key 有 UNIQUE 约束，并发 create_attempt 不重复。
  - version 乐观锁（CAS）。
  - 进程重启后 Task/Attempt/Assignment 可恢复。
  - 数据损坏时启动失败（fail closed），不静默初始化空 Store。
  - 所有磁盘 IO 通过 asyncio.to_thread，不阻塞事件循环。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.errors import CASConflictError, InvalidTransitionError
from mcc_core.runtime.task_record import AssignmentStatus, AttemptStatus
from mcc_core.runtime.task_store_sqlite import SqliteTaskStore


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "tasks.db"


@pytest.fixture
async def store(tmp_path: Path) -> SqliteTaskStore:
    s = SqliteTaskStore(_db_path(tmp_path))
    await s.init()
    return s


class TestPR07SqliteBasicCRUD:
    """基本 CRUD 操作。"""

    async def test_create_and_get_task(self, store: SqliteTaskStore):
        record = await store.create_task("subject-1", "desc-1")
        assert record.subject == "subject-1"
        assert record.description == "desc-1"
        assert record.status is TaskStatus.PENDING

        got = await store.get_task(record.task_id)
        assert got is not None
        assert got.subject == "subject-1"

    async def test_get_nonexistent_task_returns_none(
        self,
        store: SqliteTaskStore,
    ):
        assert await store.get_task("no-such-task") is None

    async def test_list_tasks_by_run_id(self, store: SqliteTaskStore):
        r1 = await store.create_task("s1", "d1", run_id="run-A")
        r2 = await store.create_task("s2", "d2", run_id="run-A")
        await store.create_task("s3", "d3", run_id="run-B")

        tasks = await store.list_tasks(run_id="run-A")
        ids = {t.task_id for t in tasks}
        assert ids == {r1.task_id, r2.task_id}

    async def test_list_tasks_by_status(self, store: SqliteTaskStore):
        r1 = await store.create_task("s1", "d1")
        r2 = await store.create_task("s2", "d2")
        await store.transition_task(
            r2.task_id, TaskStatus.PENDING, TaskStatus.DISPATCHED, agent_id="a-1",
        )

        pending = await store.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == r1.task_id

    async def test_transition_task(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        updated = await store.transition_task(
            record.task_id,
            TaskStatus.PENDING,
            TaskStatus.DISPATCHED,
            agent_id="a-1",
        )
        assert updated.status is TaskStatus.DISPATCHED
        assert updated.agent_id == "a-1"
        assert updated.version > record.version

    async def test_transition_wrong_from_status_raises(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        with pytest.raises(InvalidTransitionError):
            await store.transition_task(
                record.task_id,
                TaskStatus.DISPATCHED,  # wrong: task is PENDING
                TaskStatus.RUNNING,
            )


class TestPR07SqliteAttemptLifecycle:
    """Attempt 生命周期。"""

    async def test_create_and_get_attempt(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        assert attempt.task_id == record.task_id
        assert attempt.status is AttemptStatus.PENDING

        got = await store.get_attempt(attempt.attempt_id)
        assert got is not None
        assert got.attempt_id == attempt.attempt_id

    async def test_start_attempt(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        started = await store.start_attempt(attempt.attempt_id)
        assert started.status is AttemptStatus.RUNNING

    async def test_complete_attempt(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        await store.start_attempt(attempt.attempt_id)
        completed = await store.complete_attempt(
            attempt.attempt_id, summary="done", cost_usd=0.01,
        )
        assert completed.status is AttemptStatus.SUCCEEDED
        assert completed.summary == "done"
        assert completed.cost_usd == 0.01

    async def test_fail_attempt(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d", max_attempts=1)
        attempt = await store.create_attempt(record.task_id, "agent-1")
        await store.start_attempt(attempt.attempt_id)
        failed = await store.fail_attempt(attempt.attempt_id, error="boom")
        assert failed.status is AttemptStatus.FAILED
        assert failed.error == "boom"

    async def test_list_attempts(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        a1 = await store.create_attempt(record.task_id, "agent-1")
        a2 = await store.create_attempt(record.task_id, "agent-1")
        attempts = await store.list_attempts(record.task_id)
        ids = {a.attempt_id for a in attempts}
        assert ids == {a1.attempt_id, a2.attempt_id}


class TestPR07SqliteIdempotency:
    """幂等键 UNIQUE 约束。"""

    async def test_idempotency_key_dedup(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        a1 = await store.create_attempt(
            record.task_id, "agent-1", idempotency_key="my-key",
        )
        a2 = await store.create_attempt(
            record.task_id, "agent-1", idempotency_key="my-key",
        )
        assert a1.attempt_id == a2.attempt_id

    async def test_find_by_idempotency_key(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        await store.create_attempt(
            record.task_id, "agent-1", idempotency_key="find-me",
        )
        found = await store.find_by_idempotency_key("find-me")
        assert found is not None
        assert found.idempotency_key == "find-me"

    async def test_find_nonexistent_key_returns_none(
        self,
        store: SqliteTaskStore,
    ):
        assert await store.find_by_idempotency_key("no-such-key") is None


class TestPR07SqliteCAS:
    """CAS 终态保护。"""

    async def test_complete_after_cancel_raises_cas(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        await store.start_attempt(attempt.attempt_id)
        await store.fail_attempt(
            attempt.attempt_id, error="cancelled", status=AttemptStatus.CANCELLED,
        )
        with pytest.raises(CASConflictError):
            await store.complete_attempt(attempt.attempt_id, summary="late")

    async def test_cancel_after_complete_raises_cas(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        await store.start_attempt(attempt.attempt_id)
        await store.complete_attempt(attempt.attempt_id, summary="done")
        with pytest.raises(CASConflictError):
            await store.fail_attempt(
                attempt.attempt_id, error="late", status=AttemptStatus.CANCELLED,
            )

    async def test_duplicate_complete_is_idempotent(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        await store.start_attempt(attempt.attempt_id)
        first = await store.complete_attempt(attempt.attempt_id, summary="done")
        second = await store.complete_attempt(attempt.attempt_id, summary="again")
        assert second.status is AttemptStatus.SUCCEEDED
        assert second.version == first.version


class TestPR07SqliteAssignment:
    """Assignment 生命周期。"""

    async def test_create_and_get_assignment(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        assignment = await store.create_assignment(
            record.task_id, attempt.attempt_id, "agent-1", lease_seconds=120,
        )
        assert assignment.lease_expires_at is not None

        got = await store.get_assignment(assignment.assignment_id)
        assert got is not None
        assert got.agent_id == "agent-1"

    async def test_complete_assignment(self, store: SqliteTaskStore):
        record = await store.create_task("s", "d")
        attempt = await store.create_attempt(record.task_id, "agent-1")
        assignment = await store.create_assignment(
            record.task_id, attempt.attempt_id, "agent-1",
        )
        completed = await store.complete_assignment(assignment.assignment_id)
        assert completed.status is AssignmentStatus.COMPLETED


class TestPR07SqliteRestartRecovery:
    """进程重启后数据可恢复。"""

    async def test_task_persists_across_restart(self, tmp_path: Path):
        db = _db_path(tmp_path)
        s1 = SqliteTaskStore(db)
        await s1.init()
        record = await s1.create_task("persisted", "desc")
        await s1.close()

        s2 = SqliteTaskStore(db)
        await s2.init()
        got = await s2.get_task(record.task_id)
        assert got is not None
        assert got.subject == "persisted"
        await s2.close()

    async def test_attempt_persists_across_restart(self, tmp_path: Path):
        db = _db_path(tmp_path)
        s1 = SqliteTaskStore(db)
        await s1.init()
        record = await s1.create_task("s", "d")
        attempt = await s1.create_attempt(record.task_id, "agent-1")
        await s1.start_attempt(attempt.attempt_id)
        await s1.close()

        s2 = SqliteTaskStore(db)
        await s2.init()
        got = await s2.get_attempt(attempt.attempt_id)
        assert got is not None
        assert got.status is AttemptStatus.RUNNING
        await s2.close()

    async def test_assignment_persists_across_restart(self, tmp_path: Path):
        db = _db_path(tmp_path)
        s1 = SqliteTaskStore(db)
        await s1.init()
        record = await s1.create_task("s", "d")
        attempt = await s1.create_attempt(record.task_id, "agent-1")
        assignment = await s1.create_assignment(
            record.task_id, attempt.attempt_id, "agent-1",
        )
        await s1.close()

        s2 = SqliteTaskStore(db)
        await s2.init()
        got = await s2.get_assignment(assignment.assignment_id)
        assert got is not None
        assert got.agent_id == "agent-1"
        await s2.close()

    async def test_idempotency_index_persists_across_restart(
        self,
        tmp_path: Path,
    ):
        db = _db_path(tmp_path)
        s1 = SqliteTaskStore(db)
        await s1.init()
        record = await s1.create_task("s", "d")
        await s1.create_attempt(
            record.task_id, "agent-1", idempotency_key="restart-key",
        )
        await s1.close()

        s2 = SqliteTaskStore(db)
        await s2.init()
        found = await s2.find_by_idempotency_key("restart-key")
        assert found is not None
        await s2.close()


class TestPR07SqliteIntegrity:
    """数据损坏时 fail closed。"""

    async def test_corrupted_db_raises_on_init(self, tmp_path: Path):
        db = _db_path(tmp_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(b"not a valid sqlite database")

        store = SqliteTaskStore(db)
        with pytest.raises(RuntimeError):
            await store.init()

    async def test_schema_version_tracked(self, tmp_path: Path):
        db = _db_path(tmp_path)
        s = SqliteTaskStore(db)
        await s.init()
        await s.close()

        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
            ).fetchone()
            assert row is not None
            assert int(row[0]) >= 1
        finally:
            conn.close()


class TestPR07SqliteLeaseExpiry:
    """Lease 过期回收。"""

    async def test_expire_leases_returns_expired_tasks(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        await store.claim_task(record.task_id, "agent-1", lease_seconds=0)
        expired = await store.expire_leases()
        assert record.task_id in expired

    async def test_expire_leases_resets_task_to_pending(
        self,
        store: SqliteTaskStore,
    ):
        record = await store.create_task("s", "d")
        await store.claim_task(record.task_id, "agent-1", lease_seconds=0)
        await store.expire_leases()
        task = await store.get_task(record.task_id)
        assert task is not None
        assert task.status is TaskStatus.PENDING
