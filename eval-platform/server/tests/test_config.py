"""配置解析测试：默认值、显式覆盖与非法输入。"""

import pytest
from eval_server.config import (
    DEFAULT_CORS_ORIGINS,
    DEFAULT_DATABASE_URL,
    DEFAULT_LEASE_TTL_S,
    DEFAULT_PORT,
    ServerConfig,
)


def test_defaults_with_empty_env() -> None:
    cfg = ServerConfig.from_env({})
    assert cfg.database_url == DEFAULT_DATABASE_URL
    assert cfg.port == DEFAULT_PORT
    assert cfg.runner_secret is None
    assert cfg.cors_origins == DEFAULT_CORS_ORIGINS
    assert cfg.lease_ttl_s == DEFAULT_LEASE_TTL_S


@pytest.mark.parametrize("raw", ["8081", "9000", " 7000 "])
def test_port_from_env(raw: str) -> None:
    cfg = ServerConfig.from_env({"PORT": raw})
    assert cfg.port == int(raw)


def test_empty_port_falls_back_to_default() -> None:
    cfg = ServerConfig.from_env({"PORT": ""})
    assert cfg.port == DEFAULT_PORT


@pytest.mark.parametrize("raw", ["abc", "-1", "0", "70000", "8.5"])
def test_invalid_port_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="PORT"):
        ServerConfig.from_env({"PORT": raw})


def test_cors_origins_parsed_and_trimmed() -> None:
    cfg = ServerConfig.from_env({"CORS_ALLOWED_ORIGINS": "http://a.com, http://b.com ,,,"})
    assert cfg.cors_origins == ("http://a.com", "http://b.com")


def test_cors_origins_fallback_on_blank() -> None:
    assert ServerConfig.from_env({"CORS_ALLOWED_ORIGINS": "   "}).cors_origins == (
        DEFAULT_CORS_ORIGINS
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("s3cret", "s3cret"), ("", None), (None, None)],
)
def test_runner_secret_normalized(raw: str | None, expected: str | None) -> None:
    env = {} if raw is None else {"RUNNER_SECRET": raw}
    assert ServerConfig.from_env(env).runner_secret is expected


def test_database_url_override() -> None:
    cfg = ServerConfig.from_env({"DATABASE_URL": "postgresql://eval:pw@db:5432/eval"})
    assert cfg.database_url == "postgresql://eval:pw@db:5432/eval"


@pytest.mark.parametrize("raw", ["60", "300"])
def test_lease_ttl_from_env(raw: str) -> None:
    assert ServerConfig.from_env({"LEASE_TTL_S": raw}).lease_ttl_s == int(raw)


@pytest.mark.parametrize("raw", ["abc", "0", "-5"])
def test_invalid_lease_ttl_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="LEASE_TTL_S"):
        ServerConfig.from_env({"LEASE_TTL_S": raw})


def test_env_source_defaults_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    cfg = ServerConfig.from_env()
    assert cfg.port == 9999
