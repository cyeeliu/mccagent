"""数据库访问层测试：迁移幂等执行。"""

from pathlib import Path

from conftest import FakeDb
from eval_server.db import apply_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


async def test_apply_migrations_applies_all_in_order(fake_db: FakeDb) -> None:
    applied = await apply_migrations(fake_db, MIGRATIONS_DIR)
    assert applied == ["0001_init.sql", "0002_constraints.sql"]
    assert fake_db.migrations == ["0001_init.sql", "0002_constraints.sql"]


async def test_apply_migrations_is_idempotent(fake_db: FakeDb) -> None:
    await apply_migrations(fake_db, MIGRATIONS_DIR)
    second = await apply_migrations(fake_db, MIGRATIONS_DIR)
    assert second == []
    assert fake_db.migrations == ["0001_init.sql", "0002_constraints.sql"]


async def test_apply_migrations_skips_already_applied(fake_db: FakeDb) -> None:
    fake_db.migrations.append("0001_init.sql")
    applied = await apply_migrations(fake_db, MIGRATIONS_DIR)
    assert applied == ["0002_constraints.sql"]
