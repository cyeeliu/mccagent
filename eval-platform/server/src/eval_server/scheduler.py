"""任务调度：租约领取、心跳续租与 reaper 自愈。

核心是 PostgreSQL 的 ``FOR UPDATE SKIP LOCKED``：多个 Runner 并发领取时
每个事务只锁自己拿到的行，拿不到就静默跳过，互不阻塞、绝无双重分配；
批次并发上限的检查也放进同一条 SQL，在事务内原子完成。

run 生命周期：queued → leased → running → passed/failed/error/timeout；
租约到期未续由 reaper 回收回队，重试超过上限直接置 error。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from eval_server.db import DbLike

RUN_STATUSES: frozenset[str] = frozenset(
    {"queued", "leased", "running", "passed", "failed", "error", "timeout"}
)
ACTIVE_STATUSES: frozenset[str] = frozenset({"leased", "running"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"passed", "failed", "error", "timeout"})

BATCH_STATUSES: frozenset[str] = frozenset({"pending", "running", "done", "cancelled"})

MAX_RETRIES = 3
REAPER_INTERVAL_S = 30.0

LEASE_SELECT_SQL = """
SELECT r.id, r.batch_id, r.case_id
FROM runs r
JOIN batches b ON b.id = r.batch_id
WHERE r.status = 'queued'
  AND b.status = 'running'
  AND (SELECT count(*) FROM runs x
       WHERE x.batch_id = b.id AND x.status IN ('leased','running')) < b.parallelism
ORDER BY b.priority DESC, r.created_at
FOR UPDATE OF r SKIP LOCKED
LIMIT 1
"""

LEASE_UPDATE_SQL = """
UPDATE runs SET status = 'leased', runner_id = $2,
                 lease_expires_at = now() + make_interval(secs => $3)
WHERE id = $1
"""

LEASE_TASK_SQL = """
SELECT c->>'task' AS task,
       COALESCE(c->'expectations', '[]'::jsonb) AS expectations,
       p.scaffold, p.model
FROM batches b
JOIN agent_profiles p ON p.id = b.profile_id,
     LATERAL jsonb_array_elements(b.cases) AS c
WHERE b.id = $1 AND c->>'case_id' = $2
"""

HEARTBEAT_SQL = """
UPDATE runs SET lease_expires_at = now() + make_interval(secs => $3),
                 status = 'running',
                 started_at = COALESCE(started_at, now())
WHERE id = $1 AND runner_id = $2 AND status IN ('leased','running')
"""

REAPER_REQUEUE_SQL = """
UPDATE runs SET status = 'queued', runner_id = NULL,
                 lease_expires_at = NULL, retries = retries + 1
WHERE status IN ('leased','running') AND lease_expires_at < now()
  AND retries < $1
"""

REAPER_DEAD_SQL = """
UPDATE runs SET status = 'error',
                 error = 'lease expired without heartbeat, retries exhausted',
                 finished_at = now()
WHERE status IN ('leased','running') AND lease_expires_at < now()
"""

BATCH_DONE_SQL = """
UPDATE batches b SET status = 'done'
WHERE b.status = 'running'
  AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.batch_id = b.id
                  AND r.status IN ('queued','leased','running'))
"""

QUEUE_DEPTH_SQL = """
SELECT
  (SELECT count(*) FROM runs WHERE status = 'queued') AS queued,
  (SELECT count(*) FROM runs WHERE status IN ('leased','running')) AS active
"""


class LeasedRun(BaseModel):
    """一次任务领取的结果，Runner 据此构造执行请求。"""

    run_id: str
    batch_id: str
    case_id: str
    task: str
    scaffold: str
    model: str
    expectations: Any = []


def _rows_affected(status: str) -> int:
    """解析 asyncpg execute 返回值（如 'UPDATE 1'）中的行数。"""
    parts = status.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


async def lease_next(db: DbLike, runner_id: str, lease_ttl_s: int) -> LeasedRun | None:
    """Runner 领取下一个可执行任务；无可领任务时返回 None。"""
    async with db.acquire() as conn, conn.transaction():
        row = await conn.fetchrow(LEASE_SELECT_SQL)
        if row is None:
            return None
        await conn.execute(LEASE_UPDATE_SQL, row["id"], runner_id, lease_ttl_s)
        task_row = await conn.fetchrow(LEASE_TASK_SQL, row["batch_id"], row["case_id"])

    return LeasedRun(
        run_id=str(row["id"]),
        batch_id=str(row["batch_id"]),
        case_id=row["case_id"],
        task=task_row["task"],
        scaffold=task_row["scaffold"],
        model=task_row["model"],
        expectations=task_row["expectations"],
    )


async def heartbeat(db: DbLike, run_id: str, runner_id: str, lease_ttl_s: int) -> bool:
    """心跳续租；返回 False 表示租约已被回收，Runner 应放弃该任务。"""
    status = await db.execute(HEARTBEAT_SQL, run_id, runner_id, lease_ttl_s)
    return _rows_affected(status) == 1


async def run_reaper_once(db: DbLike) -> None:
    """执行一轮 reaper 逻辑：回收过期租约 → 回队重试或置 error，并收尾批次。"""
    await db.execute(REAPER_REQUEUE_SQL, MAX_RETRIES)
    await db.execute(REAPER_DEAD_SQL)
    await db.execute(BATCH_DONE_SQL)


async def queue_depth(db: DbLike) -> tuple[int, int]:
    """返回 (排队数, 活跃数) 快照。"""
    row = await db.fetchrow(QUEUE_DEPTH_SQL)
    return int(row["queued"]), int(row["active"])


async def reaper_loop(
    db: DbLike,
    *,
    interval_s: float = REAPER_INTERVAL_S,
    on_tick: Callable[[int, int], None] | None = None,
) -> None:
    """后台回收循环；on_tick(queued, active) 用于上报指标快照。"""
    while True:
        await asyncio.sleep(interval_s)
        await run_reaper_once(db)
        queued, active = await queue_depth(db)
        if on_tick is not None:
            on_tick(queued, active)
