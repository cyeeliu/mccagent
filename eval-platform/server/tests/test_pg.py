"""真实 PostgreSQL 集成测试。

需要可用的 PostgreSQL 实例；通过环境变量开启：

    EVAL_PG_DSN=postgres://eval:eval@localhost:5432/eval_test

未设置时整组跳过（无数据库的环境仅跑内存替身测试）。
"""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("EVAL_PG_DSN"), reason="EVAL_PG_DSN not set; PG integration skipped"
)

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


@pytest.fixture
async def pg_pool():
    import asyncpg
    from eval_server.db import apply_migrations

    pool = await asyncpg.create_pool(os.environ["EVAL_PG_DSN"])
    try:
        await apply_migrations(pool, MIGRATIONS)
        yield pool
    finally:
        await pool.close()


async def test_apply_migrations_is_idempotent_on_real_pg(pg_pool) -> None:
    from eval_server.db import apply_migrations

    applied = await apply_migrations(pg_pool, MIGRATIONS)
    assert applied == []  # fixture 已应用过


async def test_lease_and_heartbeat_on_real_pg(pg_pool) -> None:
    from eval_server.scheduler import heartbeat, lease_next

    async with pg_pool.acquire() as conn:
        profile = await conn.fetchrow(
            "INSERT INTO agent_profiles (name, scaffold, model)"
            " VALUES ('pg-it', 'mock', 'm') RETURNING id"
        )
        batch = await conn.fetchrow(
            "INSERT INTO batches (name, profile_id, cases)"
            " VALUES ('pg-it', $1, $2) RETURNING id",
            profile["id"],
            json.dumps([{"case_id": "c1", "task": "hello"}]),
        )
        await conn.execute(
            "INSERT INTO runs (batch_id, case_id) VALUES ($1, 'c1')", batch["id"]
        )

    try:
        leased = await lease_next(pg_pool, "pg-runner", 60)
        assert leased is not None
        assert leased.task == "hello"
        assert leased.scaffold == "mock"
        assert leased.case_id == "c1"

        assert await heartbeat(pg_pool, leased.run_id, "pg-runner", 60) is True
        assert await heartbeat(pg_pool, leased.run_id, "other-runner", 60) is False
    finally:
        async with pg_pool.acquire() as conn:
            await conn.execute("DELETE FROM runs WHERE batch_id = $1", batch["id"])
            await conn.execute("DELETE FROM batches WHERE id = $1", batch["id"])
            await conn.execute("DELETE FROM agent_profiles WHERE id = $1", profile["id"])
