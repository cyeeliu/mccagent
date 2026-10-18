"""CLI 入口测试：version / config / headless / sessions / resume。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from click.testing import CliRunner
from mcc_cli import main
from mcc_cli.events import EchoEventSource
from mcc_cli.headless import run_headless
from mcc_session import SessionMeta


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "mcc" in result.output
    assert "0.1.0" in result.output


def test_config_default(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model"]["main"] == "claude-opus-4-8"
    assert data["budget"]["max_usd_per_session"] == 2.0


def test_headless_outputs_response(capsys):
    source = EchoEventSource()
    rc = asyncio.run(run_headless(source, "你好", quiet=False))
    captured = capsys.readouterr()
    assert rc == 0
    assert "模拟响应" in captured.out


def test_headless_quiet_suppresses_stats(capsys):
    source = EchoEventSource()
    rc = asyncio.run(run_headless(source, "test", quiet=True))
    captured = capsys.readouterr()
    assert rc == 0
    assert "模拟响应" in captured.out
    assert "USD" not in captured.err


def test_headless_stderr_shows_tool_progress(capsys):
    source = EchoEventSource()
    asyncio.run(run_headless(source, "list files", quiet=False))
    captured = capsys.readouterr()
    assert "⚙" in captured.err


def _make_session(base: Path, sid: str, started: datetime, turns: int = 3,
                  cost: float = 0.25, title: str | None = None) -> None:
    day = base / started.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    meta = SessionMeta(
        id=sid, started_at=started, cwd="/tmp", turns=turns,
        cost_usd=cost, status="ended", title=title,
    )
    (day / f"{sid}.meta.json").write_text(
        meta.model_dump_json(), encoding="utf-8"
    )


class TestSessionsCmd:
    def test_list_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "list"])
        assert result.exit_code == 0
        assert "No sessions found." in result.output

    def test_list_shows_sessions(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        _make_session(
            home / ".mcc" / "sessions", "abc123",
            datetime(2026, 10, 1, 14, 32, tzinfo=UTC),
            turns=8, cost=0.34, title="修复 rpc client 超时",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "list"])
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "2026-10-01 14:32" in result.output
        assert "8" in result.output
        assert "$0.3400" in result.output
        assert "修复 rpc client 超时" in result.output

    def test_list_sorted_desc(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        base = home / ".mcc" / "sessions"
        _make_session(base, "older", datetime(2026, 9, 1, tzinfo=UTC))
        _make_session(base, "newer", datetime(2026, 10, 1, tzinfo=UTC))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions"])
        assert result.exit_code == 0
        assert result.output.index("newer") < result.output.index("older")

    def test_purge_removes_old(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        base = home / ".mcc" / "sessions"
        _make_session(base, "ancient", datetime(2026, 1, 1, tzinfo=UTC))
        _make_session(base, "recent", datetime.now(UTC))
        runner = CliRunner()
        result = runner.invoke(
            main, ["sessions", "purge", "--older-than", "30d"]
        )
        assert result.exit_code == 0
        assert "purged 1 session(s)" in result.output
        assert "recent" in runner.invoke(main, ["sessions", "list"]).output
        assert "ancient" not in runner.invoke(main, ["sessions", "list"]).output

    def test_delete_removes_session(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        base = home / ".mcc" / "sessions"
        _make_session(base, "victim", datetime.now(UTC), turns=3)
        _make_session(base, "survivor", datetime.now(UTC))
        runner = CliRunner()

        result = runner.invoke(main, ["sessions", "delete", "victim"])

        assert result.exit_code == 0
        assert "deleted session victim (3 turn(s))" in result.output
        listing = runner.invoke(main, ["sessions", "list"]).output
        assert "victim" not in listing
        assert "survivor" in listing

    def test_delete_missing_session_fails(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "delete", "ghost"])
        assert result.exit_code == 1
        assert "session not found: ghost" in result.output

    def test_delete_requires_id(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "delete"])
        assert result.exit_code == 2
        assert "Usage: mcc sessions delete <session-id>" in result.output

    def test_purge_requires_valid_spec(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "purge"])
        assert result.exit_code == 2
        result = runner.invoke(main, ["sessions", "purge", "--older-than", "xx"])
        assert result.exit_code == 2

    def test_unknown_subcommand(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["sessions", "bogus"])
        assert result.exit_code == 2


class TestResumeCmd:
    def test_resume_requires_id(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["resume"])
        assert result.exit_code == 2
        assert "Usage" in result.output

    def test_resume_unknown_session(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["resume", "ghost"])
        assert result.exit_code == 1
        assert "session not found" in result.output

    def test_resume_without_llm_key_errors(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _make_session(home / ".mcc" / "sessions", "r1", datetime.now(UTC))
        runner = CliRunner()
        result = runner.invoke(main, ["resume", "r1"])
        assert result.exit_code == 2
        assert "API key" in result.output


def _config(main="claude-opus-4-8", subagent=None, summarize=None):
    from mcc_config import Config, ModelConfig

    return Config(
        model=ModelConfig(main=main, subagent=subagent, summarize=summarize)
    )


class TestResolveModel:
    def test_explicit_beats_env_and_json(self, monkeypatch):
        from mcc_cli import _resolve_model

        monkeypatch.setenv("MODEL", "env-model")
        cfg = _config(summarize="json-sum")
        assert _resolve_model(cfg, "summarize", "cli-model") == "cli-model"

    def test_env_beats_json(self, monkeypatch):
        from mcc_cli import _resolve_model

        monkeypatch.setenv("MODEL", "env-model")
        cfg = _config(main="json-main", summarize="json-sum")
        assert _resolve_model(cfg, "main") == "env-model"
        assert _resolve_model(cfg, "summarize") == "env-model"
        assert _resolve_model(cfg, "subagent") == "env-model"

    def test_json_kind_when_env_unset(self, monkeypatch):
        from mcc_cli import _resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        cfg = _config(main="json-main", summarize="json-sum")
        assert _resolve_model(cfg, "summarize") == "json-sum"
        assert _resolve_model(cfg, "main") == "json-main"

    def test_unconfigured_kind_falls_back_to_main(self, monkeypatch):
        from mcc_cli import _resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        cfg = _config(main="the-main")
        assert _resolve_model(cfg, "summarize") == "the-main"
        assert _resolve_model(cfg, "subagent") == "the-main"

    def test_default_when_nothing_configured(self, monkeypatch):
        from mcc_cli import _resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        assert _resolve_model(_config(), "main") == "claude-opus-4-8"


class TestBuildSourceModelResolution:
    def test_both_models_fall_back_to_main(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("MODEL", raising=False)
        source = _build_source(tmp_path, None, True, None)
        assert source.agent.model == "claude-opus-4-8"
        assert source.agent.summarize_model == "claude-opus-4-8"

    def test_env_model_overrides_both(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("MODEL", "env-model")
        source = _build_source(tmp_path, None, True, None)
        assert source.agent.model == "env-model"
        assert source.agent.summarize_model == "env-model"

    def test_cli_model_only_overrides_main(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("MODEL", raising=False)
        source = _build_source(tmp_path, "cli-main", True, None)
        assert source.agent.model == "cli-main"
        assert source.agent.summarize_model == "claude-opus-4-8"

    def test_json_summarize_respected(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("MODEL", raising=False)
        _build_source(tmp_path, None, True, None)

        settings = tmp_path / ".mcc" / "settings.json"
        data = json.loads(settings.read_text(encoding="utf-8"))
        data["model"] = {
            "main": "proj-main",
            "subagent": None,
            "summarize": "proj-sum",
        }
        settings.write_text(json.dumps(data), encoding="utf-8")

        source = _build_source(tmp_path, None, True, None)
        assert source.agent.model == "proj-main"
        assert source.agent.summarize_model == "proj-sum"


class TestBuildSourceSubagent:
    def test_spawn_tool_registered(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("MODEL", raising=False)
        source = _build_source(tmp_path, None, True, None)
        tool = source.agent.registry.get("spawn_subagent")
        assert tool is not None
        assert set(tool.presets) == {
            "code_explorer",
            "test_runner",
            "summarizer",
        }

    def test_preset_models_fall_back_to_main(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.delenv("MODEL", raising=False)
        source = _build_source(tmp_path, None, True, None)
        tool = source.agent.registry.get("spawn_subagent")
        assert tool.presets["code_explorer"].model == "claude-opus-4-8"
        assert tool.presets["summarizer"].model == "claude-opus-4-8"

    def test_preset_models_from_json(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("MODEL", raising=False)
        _build_source(tmp_path, None, True, None)

        settings = tmp_path / ".mcc" / "settings.json"
        data = json.loads(settings.read_text(encoding="utf-8"))
        data["model"] = {
            "main": "proj-main",
            "subagent": "proj-sub",
            "summarize": "proj-sum",
        }
        settings.write_text(json.dumps(data), encoding="utf-8")

        source = _build_source(tmp_path, None, True, None)
        tool = source.agent.registry.get("spawn_subagent")
        assert tool.presets["code_explorer"].model == "proj-sub"
        assert tool.presets["summarizer"].model == "proj-sum"
