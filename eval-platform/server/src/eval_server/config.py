"""服务配置：环境变量解析与校验。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_DATABASE_URL = "postgresql://postgres:dev@localhost:5432/evalplatform"
DEFAULT_PORT = 8080
DEFAULT_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
DEFAULT_LEASE_TTL_S = 120


@dataclass(frozen=True)
class ServerConfig:
    """API 服务的全部启动参数。

    Attributes:
        database_url: PostgreSQL 连接串。
        port: HTTP 监听端口。
        runner_secret: 内部接口 Bearer token；未设置视为开发模式（放行 + 警告）。
        cors_origins: 允许的浏览器来源白名单。
        lease_ttl_s: 任务租约时长；超时未续租的 run 由 reaper 回收。
    """

    database_url: str
    port: int
    runner_secret: str | None
    cors_origins: tuple[str, ...]
    lease_ttl_s: int = DEFAULT_LEASE_TTL_S

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServerConfig:
        """从环境变量构建配置；env 参数缺省时读取进程环境。"""
        source = os.environ if env is None else env
        return cls(
            database_url=source.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
            port=_parse_port(source.get("PORT")),
            runner_secret=source.get("RUNNER_SECRET") or None,
            cors_origins=_parse_cors(source.get("CORS_ALLOWED_ORIGINS")),
            lease_ttl_s=_parse_ttl(source.get("LEASE_TTL_S")),
        )


def _parse_port(raw: str | None) -> int:
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid PORT: {raw!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"PORT out of range: {port}")
    return port


def _parse_cors(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        origins = DEFAULT_CORS_ORIGINS
    else:
        origins = tuple(o.strip() for o in raw.split(",") if o.strip())
    if not origins:
        raise ValueError("CORS_ALLOWED_ORIGINS must contain at least one origin")
    return origins


def _parse_ttl(raw: str | None) -> int:
    if not raw:
        return DEFAULT_LEASE_TTL_S
    try:
        ttl = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid LEASE_TTL_S: {raw!r}") from exc
    if ttl <= 0:
        raise ValueError(f"LEASE_TTL_S must be positive: {ttl}")
    return ttl
