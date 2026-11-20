"""SQLite 持久化任务存储：可靠的单机/单节点基线实现。

替代 JSON 文件持久化，提供：
- 事务原子性（BEGIN/COMMIT/ROLLBACK）
- idempotency_key UNIQUE 约束
- version 乐观锁（CAS）
- 进程重启后数据可恢复
- 数据损坏时 fail closed（不静默初始化空 Store）
- 所有磁盘 IO 通过 asyncio.to_thread，不阻塞事件循环
- schema version 追踪与 integrity check
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.errors import (
    AttemptNotFoundError,
    CASConflictError,
    InvalidTransitionError,
    TaskNotFoundError,
)
from mcc_core.runtime.identifiers import AgentId, RunId, RuntimeId, TaskId
from mcc_core.runtime.result import TokenUsage
from mcc_core.runtime.task_record import (
    AssignmentStatus,
    AttemptStatus,
    TaskAssignment,
    TaskAttempt,
    TaskRecord,
)

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id          TEXT PRIMARY KEY,
    subject          TEXT NOT NULL,
    description      TEXT NOT NULL,
    status           TEXT NOT NULL,
    agent_id         TEXT,
    run_id           TEXT,
    blocked_by_json  TEXT NOT NULL DEFAULT '[]',
    priority         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    attempt_count    INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    deadline_at      TEXT,
    lease_expires_at TEXT,
    error            TEXT,
    result_ref       TEXT,
    version          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id      TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    runtime_id      TEXT,
    status          TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    error           TEXT,
    summary         TEXT,
    output_json     TEXT,
    usage_json      TEXT,
    cost_usd        REAL,
    artifacts_json  TEXT,
    version         INTEGER NOT NULL DEFAULT 0,
    UNIQUE(idempotency_key)
);

CREATE TABLE IF NOT EXISTS assignments (
    assignment_id   TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    attempt_id      TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    runtime_id      TEXT,
    status          TEXT NOT NULL,
    assigned_at     TEXT NOT NULL,
    lease_expires_at TEXT,
    completed_at    TEXT,
    version         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_attempts_task_id ON attempts(task_id);
CREATE INDEX IF NOT EXISTS idx_assignments_task_id ON assignments(task_id);
CREATE INDEX IF NOT EXISTS idx_assignments_agent_id ON assignments(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_run_id ON tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _json_dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def _json_loads(s: str | None) -> Any:
    if s is None:
        return None
    return json.loads(s)


def _usage_to_str(usage: TokenUsage | None) -> str | None:
    if usage is None:
        return None
    return json.dumps(usage.to_dict(), ensure_ascii=False)


def _str_to_usage(s: str | None) -> TokenUsage | None:
    if s is None:
        return None
    d = json.loads(s)
    return TokenUsage(
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        cache_creation_input_tokens=d.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=d.get("cache_read_input_tokens", 0),
    )


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        task_id=row["task_id"],
        subject=row["subject"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        agent_id=row["agent_id"],
        run_id=row["run_id"],
        blocked_by=set(_json_loads(row["blocked_by_json"]) or []),
        priority=row["priority"],
        max_attempts=row["max_attempts"],
        attempt_count=row["attempt_count"],
        created_at=_str_to_dt(row["created_at"]),
        updated_at=_str_to_dt(row["updated_at"]),
        deadline_at=_str_to_dt(row["deadline_at"]),
        lease_expires_at=_str_to_dt(row["lease_expires_at"]),
        error=row["error"],
        result_ref=row["result_ref"],
        version=row["version"],
    )


def _row_to_attempt(row: sqlite3.Row) -> TaskAttempt:
    return TaskAttempt(
        attempt_id=row["attempt_id"],
        task_id=row["task_id"],
        agent_id=row["agent_id"],
        runtime_id=row["runtime_id"],
        status=AttemptStatus(row["status"]),
        idempotency_key=row["idempotency_key"],
        started_at=_str_to_dt(row["started_at"]),
        finished_at=_str_to_dt(row["finished_at"]),
        error=row["error"],
        summary=row["summary"],
        output=_json_loads(row["output_json"]),
        usage=_str_to_usage(row["usage_json"]),
        cost_usd=row["cost_usd"],
        artifacts=_json_loads(row["artifacts_json"]),
        version=row["version"],
    )


def _row_to_assignment(row: sqlite3.Row) -> TaskAssignment:
    return TaskAssignment(
        assignment_id=row["assignment_id"],
        task_id=row["task_id"],
        attempt_id=row["attempt_id"],
        agent_id=row["agent_id"],
        runtime_id=row["runtime_id"],
        status=AssignmentStatus(row["status"]),
        assigned_at=_str_to_dt(row["assigned_at"]),
        lease_expires_at=_str_to_dt(row["lease_expires_at"]),
        completed_at=_str_to_dt(row["completed_at"]),
        version=row["version"],
    )


class SqliteTaskStore:
    """SQLite 持久化任务存储，实现 TaskStoreProtocol。

    线程安全：threading.Lock 保护跨线程的连接访问。
    事务：终态更新使用 BEGIN/COMMIT 保证原子性。
    幂等：idempotency_key 有 UNIQUE 约束。
    CAS：version 列用于乐观锁。
    恢复：进程重启后从 SQLite 文件恢复全部数据。
    Fail closed：损坏文件导致 init() 抛异常，不静默初始化空 Store。
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    async def init(self) -> None:
        """初始化数据库连接和 schema。

        损坏文件或 integrity check 失败时抛异常（fail closed）。
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        def _init_sync() -> None:
            try:
                conn = sqlite3.connect(
                    str(self._db_path),
                    check_same_thread=False,
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                # integrity check — fail closed
                result = conn.execute("PRAGMA integrity_check").fetchone()
                if result[0] != "ok":
                    conn.close()
                    raise RuntimeError(
                        f"SQLite integrity check failed: {result[0]}",
                    )
            except sqlite3.DatabaseError as exc:
                raise RuntimeError(
                    f"SQLite database error during init: {exc}",
                ) from exc
            conn.executescript(_SCHEMA_SQL)
            # schema version
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
            self._conn = conn

        await asyncio.to_thread(_init_sync)

    async def close(self) -> None:
        def _close_sync() -> None:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

        await asyncio.to_thread(_close_sync)

    def _conn_or_raise(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteTaskStore 未初始化，请先调用 init()")
        return self._conn

    # ------------------------------------------------------------------
    # Task 操作
    # ------------------------------------------------------------------

    async def create_task(
        self,
        subject: str,
        description: str,
        *,
        run_id: RunId | None = None,
        blocked_by: set[TaskId] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            conn = self._conn_or_raise()
            now = _now()
            task_id = f"task_{uuid4().hex[:12]}"
            record = TaskRecord(
                task_id=task_id,
                subject=subject,
                description=description,
                run_id=run_id,
                blocked_by=blocked_by or set(),
                priority=priority,
                max_attempts=max_attempts,
                created_at=now,
                updated_at=now,
            )
            with self._lock:
                conn.execute(
                    """INSERT INTO tasks
                       (task_id, subject, description, status, agent_id, run_id,
                        blocked_by_json, priority, max_attempts, attempt_count,
                        created_at, updated_at, deadline_at, lease_expires_at,
                        error, result_ref, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.task_id,
                        record.subject,
                        record.description,
                        record.status.value,
                        record.agent_id,
                        record.run_id,
                        _json_dumps(list(record.blocked_by)),
                        record.priority,
                        record.max_attempts,
                        record.attempt_count,
                        _dt_to_str(record.created_at),
                        _dt_to_str(record.updated_at),
                        _dt_to_str(record.deadline_at),
                        _dt_to_str(record.lease_expires_at),
                        record.error,
                        record.result_ref,
                        record.version,
                    ),
                )
            return record

        return await asyncio.to_thread(_sync)

    async def get_task(self, task_id: TaskId) -> TaskRecord | None:
        def _sync() -> TaskRecord | None:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            return _row_to_task(row) if row else None

        return await asyncio.to_thread(_sync)

    async def list_tasks(
        self,
        *,
        run_id: RunId | None = None,
        status: TaskStatus | None = None,
    ) -> list[TaskRecord]:
        def _sync() -> list[TaskRecord]:
            conn = self._conn_or_raise()
            clauses: list[str] = []
            params: list[Any] = []
            if run_id is not None:
                clauses.append("run_id = ?")
                params.append(run_id)
            if status is not None:
                clauses.append("status = ?")
                params.append(status.value)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._lock:
                rows = conn.execute(
                    f"SELECT * FROM tasks{where} ORDER BY created_at",
                    params,
                ).fetchall()
            return [_row_to_task(r) for r in rows]

        return await asyncio.to_thread(_sync)

    async def transition_task(
        self,
        task_id: TaskId,
        from_status: TaskStatus,
        to_status: TaskStatus,
        *,
        agent_id: AgentId | None = None,
    ) -> TaskRecord:
        def _sync() -> TaskRecord:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(task_id)
                record = _row_to_task(row)
                if record.status is not from_status:
                    raise InvalidTransitionError(
                        record.status.value,
                        to_status.value,
                    )
                now = _now()
                new_version = record.version + 1
                conn.execute(
                    """UPDATE tasks
                       SET status = ?, agent_id = ?, updated_at = ?, version = ?
                       WHERE task_id = ? AND version = ?""",
                    (
                        to_status.value,
                        agent_id or record.agent_id,
                        _dt_to_str(now),
                        new_version,
                        task_id,
                        record.version,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                return _row_to_task(row)

        return await asyncio.to_thread(_sync)

    async def claim_task(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        from datetime import timedelta

        def _sync() -> TaskRecord:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(task_id)
                record = _row_to_task(row)
                if record.status is not TaskStatus.PENDING:
                    raise InvalidTransitionError(
                        record.status.value,
                        TaskStatus.DISPATCHED.value,
                    )
                now = _now()
                lease = now + timedelta(seconds=lease_seconds)
                new_version = record.version + 1
                conn.execute(
                    """UPDATE tasks
                       SET status = ?, agent_id = ?, lease_expires_at = ?,
                           updated_at = ?, version = ?
                       WHERE task_id = ? AND version = ?""",
                    (
                        TaskStatus.DISPATCHED.value,
                        agent_id,
                        _dt_to_str(lease),
                        _dt_to_str(now),
                        new_version,
                        task_id,
                        record.version,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                return _row_to_task(row)

        return await asyncio.to_thread(_sync)

    async def renew_lease(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        lease_seconds: float = 300,
    ) -> TaskRecord:
        from datetime import timedelta

        def _sync() -> TaskRecord:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(task_id)
                record = _row_to_task(row)
                now = _now()
                lease = now + timedelta(seconds=lease_seconds)
                new_version = record.version + 1
                conn.execute(
                    """UPDATE tasks
                       SET lease_expires_at = ?, updated_at = ?, version = ?
                       WHERE task_id = ? AND version = ?""",
                    (
                        _dt_to_str(lease),
                        _dt_to_str(now),
                        new_version,
                        task_id,
                        record.version,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                return _row_to_task(row)

        return await asyncio.to_thread(_sync)

    async def expire_leases(self) -> list[TaskId]:
        def _sync() -> list[TaskId]:
            conn = self._conn_or_raise()
            now = _now()
            now_str = _dt_to_str(now)
            with self._lock:
                rows = conn.execute(
                    """SELECT * FROM tasks
                       WHERE lease_expires_at IS NOT NULL
                         AND lease_expires_at < ?
                         AND status IN ('dispatched', 'accepted', 'running')""",
                    (now_str,),
                ).fetchall()
                expired: list[TaskId] = []
                for row in rows:
                    record = _row_to_task(row)
                    new_version = record.version + 1
                    conn.execute(
                        """UPDATE tasks
                           SET status = 'pending', agent_id = NULL,
                               lease_expires_at = NULL, updated_at = ?, version = ?
                           WHERE task_id = ? AND version = ?""",
                        (
                            now_str,
                            new_version,
                            record.task_id,
                            record.version,
                        ),
                    )
                    expired.append(record.task_id)
                return expired

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Attempt 操作
    # ------------------------------------------------------------------

    async def create_attempt(
        self,
        task_id: TaskId,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        idempotency_key: str | None = None,
    ) -> TaskAttempt:
        def _sync() -> TaskAttempt:
            conn = self._conn_or_raise()
            key = idempotency_key or ""
            with self._lock:
                if key:
                    row = conn.execute(
                        "SELECT * FROM attempts WHERE idempotency_key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None:
                        return _row_to_attempt(row)
                attempt_id = f"att_{uuid4().hex[:12]}"
                actual_key = key or f"{task_id}:{attempt_id}"
                now = _now()
                attempt = TaskAttempt(
                    attempt_id=attempt_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    runtime_id=runtime_id,
                    idempotency_key=actual_key,
                    started_at=now,
                )
                conn.execute(
                    """INSERT INTO attempts
                       (attempt_id, task_id, agent_id, runtime_id, status,
                        idempotency_key, started_at, finished_at, error, summary,
                        output_json, usage_json, cost_usd, artifacts_json, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        attempt.attempt_id,
                        attempt.task_id,
                        attempt.agent_id,
                        attempt.runtime_id,
                        attempt.status.value,
                        attempt.idempotency_key,
                        _dt_to_str(attempt.started_at),
                        _dt_to_str(attempt.finished_at),
                        attempt.error,
                        attempt.summary,
                        _json_dumps(attempt.output),
                        _usage_to_str(attempt.usage),
                        attempt.cost_usd,
                        _json_dumps(attempt.artifacts),
                        attempt.version,
                    ),
                )
                # 递增 task 的 attempt_count
                conn.execute(
                    """UPDATE tasks
                       SET attempt_count = attempt_count + 1, version = version + 1
                       WHERE task_id = ?""",
                    (task_id,),
                )
                return attempt

        return await asyncio.to_thread(_sync)

    async def get_attempt(self, attempt_id: str) -> TaskAttempt | None:
        def _sync() -> TaskAttempt | None:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
            return _row_to_attempt(row) if row else None

        return await asyncio.to_thread(_sync)

    async def start_attempt(self, attempt_id: str) -> TaskAttempt:
        def _sync() -> TaskAttempt:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                if row is None:
                    raise AttemptNotFoundError(attempt_id)
                attempt = _row_to_attempt(row)
                if attempt.status is not AttemptStatus.PENDING:
                    raise InvalidTransitionError(
                        attempt.status.value,
                        AttemptStatus.RUNNING.value,
                    )
                new_version = attempt.version + 1
                conn.execute(
                    """UPDATE attempts
                       SET status = ?, version = ?
                       WHERE attempt_id = ? AND version = ?""",
                    (
                        AttemptStatus.RUNNING.value,
                        new_version,
                        attempt_id,
                        attempt.version,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                return _row_to_attempt(row)

        return await asyncio.to_thread(_sync)

    async def list_attempts(self, task_id: TaskId) -> list[TaskAttempt]:
        def _sync() -> list[TaskAttempt]:
            conn = self._conn_or_raise()
            with self._lock:
                rows = conn.execute(
                    "SELECT * FROM attempts WHERE task_id = ? ORDER BY started_at",
                    (task_id,),
                ).fetchall()
            return [_row_to_attempt(r) for r in rows]

        return await asyncio.to_thread(_sync)

    async def complete_attempt(
        self,
        attempt_id: str,
        *,
        summary: str | None = None,
        output: dict[str, object] | None = None,
        usage: TokenUsage | None = None,
        cost_usd: float | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> TaskAttempt:
        def _sync() -> TaskAttempt:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                if row is None:
                    raise AttemptNotFoundError(attempt_id)
                attempt = _row_to_attempt(row)
                # CAS 终态保护
                if attempt.is_terminal:
                    if attempt.status is AttemptStatus.SUCCEEDED:
                        return attempt  # 幂等
                    raise CASConflictError(
                        attempt_id, attempt.version, attempt.version,
                    )
                now = _now()
                new_version = attempt.version + 1
                conn.execute(
                    """UPDATE attempts
                       SET status = ?, summary = ?, output_json = ?,
                           usage_json = ?, cost_usd = ?, artifacts_json = ?,
                           finished_at = ?, version = ?
                       WHERE attempt_id = ? AND version = ?""",
                    (
                        AttemptStatus.SUCCEEDED.value,
                        summary,
                        _json_dumps(output),
                        _usage_to_str(usage),
                        cost_usd,
                        _json_dumps(artifacts),
                        _dt_to_str(now),
                        new_version,
                        attempt_id,
                        attempt.version,
                    ),
                )
                # 同步更新 TaskRecord 终态
                self._mark_task_terminal(conn, attempt.task_id, "succeeded")
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                return _row_to_attempt(row)

        return await asyncio.to_thread(_sync)

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> TaskAttempt:
        def _sync() -> TaskAttempt:
            conn = self._conn_or_raise()
            if status not in (
                AttemptStatus.FAILED,
                AttemptStatus.TIMED_OUT,
                AttemptStatus.CANCELLED,
            ):
                raise InvalidTransitionError(
                    AttemptStatus.RUNNING.value,
                    status.value,
                )
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                if row is None:
                    raise AttemptNotFoundError(attempt_id)
                attempt = _row_to_attempt(row)
                # CAS 终态保护
                if attempt.is_terminal:
                    if attempt.status is status:
                        return attempt  # 幂等
                    raise CASConflictError(
                        attempt_id, attempt.version, attempt.version,
                    )
                now = _now()
                new_version = attempt.version + 1
                conn.execute(
                    """UPDATE attempts
                       SET status = ?, error = ?, finished_at = ?, version = ?
                       WHERE attempt_id = ? AND version = ?""",
                    (
                        status.value,
                        error,
                        _dt_to_str(now),
                        new_version,
                        attempt_id,
                        attempt.version,
                    ),
                )
                # 同步更新 TaskRecord 终态
                self._mark_task_terminal(conn, attempt.task_id, "failed", error)
                row = conn.execute(
                    "SELECT * FROM attempts WHERE attempt_id = ?",
                    (attempt_id,),
                ).fetchone()
                return _row_to_attempt(row)

        return await asyncio.to_thread(_sync)

    async def find_by_idempotency_key(
        self,
        key: str,
    ) -> TaskAttempt | None:
        def _sync() -> TaskAttempt | None:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM attempts WHERE idempotency_key = ?",
                    (key,),
                ).fetchone()
            return _row_to_attempt(row) if row else None

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Assignment 操作
    # ------------------------------------------------------------------

    async def create_assignment(
        self,
        task_id: TaskId,
        attempt_id: str,
        agent_id: AgentId,
        *,
        runtime_id: RuntimeId | None = None,
        lease_seconds: float = 300,
    ) -> TaskAssignment:
        from datetime import timedelta

        def _sync() -> TaskAssignment:
            conn = self._conn_or_raise()
            now = _now()
            lease = now + timedelta(seconds=lease_seconds)
            assignment_id = f"asg_{uuid4().hex[:12]}"
            assignment = TaskAssignment(
                assignment_id=assignment_id,
                task_id=task_id,
                attempt_id=attempt_id,
                agent_id=agent_id,
                runtime_id=runtime_id,
                assigned_at=now,
                lease_expires_at=lease,
            )
            with self._lock:
                conn.execute(
                    """INSERT INTO assignments
                       (assignment_id, task_id, attempt_id, agent_id, runtime_id,
                        status, assigned_at, lease_expires_at, completed_at, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        assignment.assignment_id,
                        assignment.task_id,
                        assignment.attempt_id,
                        assignment.agent_id,
                        assignment.runtime_id,
                        assignment.status.value,
                        _dt_to_str(assignment.assigned_at),
                        _dt_to_str(assignment.lease_expires_at),
                        _dt_to_str(assignment.completed_at),
                        assignment.version,
                    ),
                )
            return assignment

        return await asyncio.to_thread(_sync)

    async def get_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment | None:
        def _sync() -> TaskAssignment | None:
            conn = self._conn_or_raise()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM assignments WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()
            return _row_to_assignment(row) if row else None

        return await asyncio.to_thread(_sync)

    async def list_assignments(
        self,
        *,
        task_id: TaskId | None = None,
        agent_id: AgentId | None = None,
        status: AssignmentStatus | None = None,
    ) -> list[TaskAssignment]:
        def _sync() -> list[TaskAssignment]:
            conn = self._conn_or_raise()
            clauses: list[str] = []
            params: list[Any] = []
            if task_id is not None:
                clauses.append("task_id = ?")
                params.append(task_id)
            if agent_id is not None:
                clauses.append("agent_id = ?")
                params.append(agent_id)
            if status is not None:
                clauses.append("status = ?")
                params.append(status.value)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._lock:
                rows = conn.execute(
                    f"SELECT * FROM assignments{where} ORDER BY assigned_at",
                    params,
                ).fetchall()
            return [_row_to_assignment(r) for r in rows]

        return await asyncio.to_thread(_sync)

    async def complete_assignment(
        self,
        assignment_id: str,
    ) -> TaskAssignment:
        def _sync() -> TaskAssignment:
            conn = self._conn_or_raise()
            now = _now()
            with self._lock:
                row = conn.execute(
                    "SELECT * FROM assignments WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()
                if row is None:
                    raise AttemptNotFoundError(assignment_id)
                assignment = _row_to_assignment(row)
                new_version = assignment.version + 1
                conn.execute(
                    """UPDATE assignments
                       SET status = ?, completed_at = ?, version = ?
                       WHERE assignment_id = ? AND version = ?""",
                    (
                        AssignmentStatus.COMPLETED.value,
                        _dt_to_str(now),
                        new_version,
                        assignment_id,
                        assignment.version,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM assignments WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()
                return _row_to_assignment(row)

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _mark_task_terminal(
        self,
        conn: sqlite3.Connection,
        task_id: TaskId,
        terminal: str,
        error: str | None = None,
    ) -> None:
        """同步更新 TaskRecord 终态（在调用方锁内执行）。"""
        row = conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return
        record = _row_to_task(row)
        status_map = {
            "succeeded": TaskStatus.SUCCEEDED.value,
            "failed": TaskStatus.FAILED.value,
        }
        new_status = status_map.get(terminal)
        if new_status is None:
            return
        if terminal == "failed" and record.can_retry:
            return  # 仍可重试，不标记终态
        if record.status.value == new_status:
            return  # 已是目标状态
        now = _now()
        new_version = record.version + 1
        conn.execute(
            """UPDATE tasks
               SET status = ?, error = ?, updated_at = ?, version = ?
               WHERE task_id = ? AND version = ?""",
            (
                new_status,
                error,
                _dt_to_str(now),
                new_version,
                task_id,
                record.version,
            ),
        )


__all__ = ["SqliteTaskStore"]
