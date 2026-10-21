"""统一模型解析与 provider 构建测试：mcc-config 是模型配置的唯一入口。

CLI 与 mcc-eval 共用 resolve_model / build_llm_provider，这里钉住
解析优先级与网关约定，防止两处实现漂移。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcc_config import DEFAULT_MODEL, build_llm_provider, resolve_model
from mcc_config.models import Config, ModelConfig


class _Settings:
    def __init__(self, main: str) -> None:
        self.model = ModelConfig(main=main)


def _write_settings(project: Path, main: str) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / ".mcc" / "settings.json").parent.mkdir(parents=True, exist_ok=True)
    (project / ".mcc" / "settings.json").write_text(
        json.dumps({"model": {"main": main}}), encoding="utf-8"
    )


# ─── resolve_model 优先级 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("explicit", "env_model", "configured", "expected"),
    [
        ("explicit-model", None, "from-config", "explicit-model"),
        (None, "from-env", "from-config", "from-env"),
        (None, None, "from-config", "from-config"),
        ("explicit-model", "from-env", "from-config", "explicit-model"),
    ],
    ids=["explicit-wins", "env-beats-config", "config-used", "explicit-beats-env"],
)
def test_resolve_model_priority(explicit, env_model, configured, expected,
                                tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MODEL", raising=False)
    if env_model:
        monkeypatch.setenv("MODEL", env_model)
    config = Config(model=ModelConfig(main=configured))

    assert resolve_model("main", explicit, config=config) == expected


def test_resolve_model_kind_falls_back_to_main(tmp_path) -> None:
    """配置只写 main 时，subagent / summarize 回落 main。"""
    config = Config(model=ModelConfig(main="main-model"))

    assert resolve_model("subagent", config=config) == "main-model"
    assert resolve_model("summarize", config=config) == "main-model"


def test_resolve_model_kind_configured_beats_main(tmp_path) -> None:
    config = Config(model=ModelConfig(main="main-model", subagent="sub-model"))

    assert resolve_model("subagent", config=config) == "sub-model"


def test_resolve_model_loads_from_work_dir(tmp_path, monkeypatch) -> None:
    """未传 config 时从 work_dir 加载合并配置。"""
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    project = tmp_path / "proj"
    _write_settings(project, "loaded-model")

    assert resolve_model("main", work_dir=project) == "loaded-model"


def test_resolve_model_default_when_all_missing(tmp_path, monkeypatch) -> None:
    """home 与 project 配置均缺失时回落 DEFAULT_MODEL（评测无人值守容错）。"""
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))

    assert resolve_model("main", work_dir=tmp_path) == DEFAULT_MODEL
    assert ModelConfig().main == DEFAULT_MODEL


# ─── build_llm_provider ──────────────────────────────────────────────────


def test_provider_explicit_base_url_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCC_LLM_BASE_URL", "http://from-env:8000/v1")
    provider = build_llm_provider(base_url="http://explicit:9000/v1")
    assert type(provider).__name__ == "OpenAiClient"
    assert provider._base_url == "http://explicit:9000/v1"


def test_provider_env_base_url_uses_openai_client(monkeypatch) -> None:
    monkeypatch.setenv("MCC_LLM_BASE_URL", "http://vllm:8000/v1")
    provider = build_llm_provider()
    assert type(provider).__name__ == "OpenAiClient"
    assert provider._base_url == "http://vllm:8000/v1"


def test_provider_auto_detects_anthropic(monkeypatch) -> None:
    monkeypatch.delenv("MCC_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    provider = build_llm_provider()
    assert type(provider).__name__ == "AnthropicClient"


def test_provider_raises_without_any_credentials(monkeypatch) -> None:
    for key in ("MCC_LLM_BASE_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError):
        build_llm_provider()
