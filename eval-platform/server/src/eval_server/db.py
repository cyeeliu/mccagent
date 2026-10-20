"""数据库访问抽象与迁移执行。

定义 DbLike / ConnLike 协议：生产环境传入 asyncpg 连接池，
测试环境传入内存实现，HTTP 层与调度器只依赖协议方法。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import asyncpg

_MIGRATION_FILE_RE = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")


def _read_pending_migrations(
    migrations_dir: str | Path, applied: set[str]
) -> list[tuple[str, str]]:
    """同步读取待应用的迁移文件，返回 (文件名, SQL) 列表。"""
    directory = Path(migrations_dir)
    files = sorted(
        p for p in directory.glob("*.sql") if _MIGRATION_FILE_RE.match(p.name)
    )
    return [(p.name, p.read_text(encoding="utf-8")) for p in files if p.name not in applied]


class ConnLike(Protocol):
    """连接级查询接口；asyncpg.Connection 与测试替身均满足。"""

    def transaction(self) -> Any: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchrow(self, query: str, *args: Any) -> Any: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def execute(self, query: str, *args: Any) -> str: ...

    async def executemany(
        self, query: str, args_seq: Iterable[Sequence[Any]]
    ) -> None: ...


class DbLike(Protocol):
    """池级接口：既能直接执行查询，也能借出连接开事务。"""

    def acquire(self) -> Any: ...

    async def fetch(self, query: str, *args: Any) -> list[Any]: ...

    async def fetchrow(self, query: str, *args: Any) -> Any: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def execute(self, query: str, *args: Any) -> str: ...


async def create_pool(database_url: str, *, min_size: int = 2, max_size: int = 16) -> asyncpg.Pool:
    """建立连接池；调用方负责关闭。"""
    return await asyncpg.create_pool(database_url, min_size=min_size, max_size=max_size)


async def apply_migrations(db: DbLike, migrations_dir: str | Path) -> list[str]:
    """按文件名序执行 migrations/*.sql，已应用的跳过。

    通过 schema_migrations 表记录进度，重复启动幂等。
    返回本次实际应用的迁移文件名列表。
    """
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
    )
    applied = {r["name"] for r in await db.fetch("SELECT name FROM schema_migrations")}
    pending = await asyncio.to_thread(_read_pending_migrations, migrations_dir, applied)

    newly_applied: list[str] = []
    for name, sql in pending:
        async with db.acquire() as conn, conn.transaction():
            await conn.execute(sql)
            await conn.execute("INSERT INTO schema_migrations (name) VALUES ($1)", name)
        newly_applied.append(name)
    return newly_applied
