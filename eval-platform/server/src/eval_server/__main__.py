"""服务启动入口：python -m eval_server。

职责仅限启动编排——读配置、连库、迁移、初始化轨迹存储与指标，
然后交给 uvicorn 运行。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import uvicorn

from eval_server.app import create_app
from eval_server.config import ServerConfig
from eval_server.db import apply_migrations, create_pool
from eval_server.state import AppState
from eval_server.store import trace_store_from_env

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def main() -> None:
    config = ServerConfig.from_env()
    if config.runner_secret is None:
        print(
            "warning: RUNNER_SECRET not set — /internal routes are"
            " unauthenticated (dev mode)",
            file=sys.stderr,
        )

    async def start() -> None:
        pool = await create_pool(config.database_url)
        try:
            applied = await apply_migrations(pool, MIGRATIONS_DIR)
            if applied:
                print(f"applied migrations: {', '.join(applied)}", file=sys.stderr)
        except Exception as exc:
            await pool.close()
            raise SystemExit(f"database migration failed: {exc}") from exc

        state = AppState(
            db=pool,
            traces=trace_store_from_env(),
            lease_ttl_s=config.lease_ttl_s,
            runner_secret=config.runner_secret,
        )
        app = create_app(state, cors_origins=config.cors_origins)
        server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=config.port, log_level="info")
        )
        try:
            await server.serve()
        finally:
            await pool.close()

    asyncio.run(start())


if __name__ == "__main__":
    main()
