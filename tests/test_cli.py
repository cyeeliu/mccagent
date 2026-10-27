"""CLI 入口测试：version / config / headless / sessions / resume。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
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


def test_config_default(tmp_path: Path, monkeypatch):
    # 隔离 home 配置与 .env 注入的 MODEL：本测试钉的是「全空环境」默认
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MODEL", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model"]["main"] == "claude-opus-4-8"
    assert data["budget"]["max_usd_per_session"] == 2.0


class TestOnboarding:
    def test_headless_without_key_exits_with_guide(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MCC_LLM_BASE_URL", raising=False)
        # settings.json 的 llm 段也算凭证来源；隔离 HOME 排除宿主配置干扰
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        runner = CliRunner()
        result = runner.invoke(main, ["-p", "hi", "--cwd", str(tmp_path)])
        assert result.exit_code == 2
        assert "欢迎使用" in result.output
        assert "ANTHROPIC_API_KEY" in result.output
        assert "--mock" in result.output

    def test_home_settings_key_counts_as_llm_source(self, tmp_path, monkeypatch):
        """home settings 里配置过 llm.api_key 时无需 env 也可启动。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MCC_LLM_BASE_URL", raising=False)
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        from mcc_config import update_home_setting

        update_home_setting(["llm", "api_key"], "sk-test")
        runner = CliRunner()
        result = runner.invoke(main, ["-p", "hi", "--cwd", str(tmp_path)])
        # 不再打引导页；构建 provider 失败也应是 LLM 错误而非 onboarding
        assert "欢迎使用 mini-claude-code" not in result.output

    def test_resume_without_key_shows_guide(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _make_session(home / ".mcc" / "sessions", "r1", datetime.now(UTC))
        runner = CliRunner()
        result = runner.invoke(main, ["resume", "r1", "--cwd", str(tmp_path)])
        assert result.exit_code == 2
        assert "欢迎使用" in result.output

    def test_base_url_counts_as_llm_source(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["-p", "hi", "--cwd", str(tmp_path), "--llm-base-url", "http://gw:8000/v1"],
        )
        assert "欢迎使用" not in result.output


class TestLlmBaseUrl:
    def test_explicit_base_url_uses_openai_compatible_client(
        self, tmp_path: Path, monkeypatch
    ):
        from mcc_cli import _build_source
        from mcc_llm.openai import OpenAiClient

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(
            tmp_path, None, False, None, llm_base_url="http://gw.internal:8000/v1"
        )
        assert isinstance(source.agent.llm, OpenAiClient)
        assert source.agent.llm._base_url == "http://gw.internal:8000/v1"

    def test_base_url_env_var_wired(self, tmp_path: Path, monkeypatch):
        """MCC_LLM_BASE_URL 环境变量经 click option 注入（声明性行为，验证元数据）。"""
        import mcc_cli

        opt = next(
            p for p in main.params if p.name == "llm_base_url"
        )
        assert opt.envvar == "MCC_LLM_BASE_URL"
        assert mcc_cli.__version__

    def test_mock_still_wins_over_base_url(self, tmp_path: Path, monkeypatch):
        from mcc_cli import _build_source
        from mcc_llm.mock import MockLlm

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(
            tmp_path, None, True, None, llm_base_url="http://gw:8000/v1"
        )
        assert isinstance(source.agent.llm, MockLlm)


class TestCompletionCmd:
    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_generates_script(self, shell):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", shell])
        assert result.exit_code == 0
        assert "mcc" in result.output
        assert len(result.output) > 50

    def test_unsupported_shell(self):
        runner = CliRunner()
        result = runner.invoke(main, ["completion", "powershell"])
        assert result.exit_code == 2
        assert "Unsupported shell" in result.output

    def test_requires_shell(self):
        runner = CliRunner()
        result = runner.invoke(main, ["completion"])
        assert result.exit_code == 2
        assert "Usage: mcc completion" in result.output


class TestVersionCompare:
    @pytest.mark.parametrize(
        ("latest", "current", "expected"),
        [
            ("0.2.0", "0.1.0", True),
            ("0.1.0", "0.1.0", False),
            ("0.1.0", "0.2.0", False),
            ("1.0.0", "0.9.9", True),
            ("0.10.0", "0.9.0", True),
            ("0.1.1", "0.1.0", True),
        ],
    )
    def test_is_newer_version(self, latest, current, expected):
        from mcc_cli import _is_newer_version

        assert _is_newer_version(latest, current) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1.2.3", (1, 2, 3)),
            ("v0.1.0", (0, 1, 0)),
            ("0.1.0rc1", (0, 1, 0)),
        ],
    )
    def test_parse_version(self, raw, expected):
        from mcc_cli import _parse_version

        assert _parse_version(raw) == expected


class TestUpdateCmd:
    def test_newer_version_advertised(self, monkeypatch):
        from mcc_cli import _latest_pypi_version

        monkeypatch.setattr(
            "mcc_cli._latest_pypi_version", lambda package="mcc-cli": "99.0.0"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "New version available: 99.0.0" in result.output
        assert "uv tool upgrade mcc-cli" in result.output

    def test_latest_version(self, monkeypatch):
        monkeypatch.setattr(
            "mcc_cli._latest_pypi_version", lambda package="mcc-cli": "0.0.1"
        )
        runner = CliRunner()
        result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "You are on the latest version." in result.output

    def test_network_failure_is_soft(self, monkeypatch):
        monkeypatch.setattr(
            "mcc_cli._latest_pypi_version", lambda package="mcc-cli": None
        )
        runner = CliRunner()
        result = runner.invoke(main, ["update"])
        assert result.exit_code == 0
        assert "Could not check for updates" in result.output

    def test_pypi_fetch_parses_version(self, monkeypatch):
        import httpx
        from mcc_cli import _latest_pypi_version

        class FakeResp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"info": {"version": "1.2.3"}}

        def fake_get(url, timeout=None):
            assert url == "https://pypi.org/pypi/mcc-cli/json"
            return FakeResp()

        monkeypatch.setattr(httpx, "get", fake_get)
        assert _latest_pypi_version() == "1.2.3"

    def test_pypi_fetch_error_returns_none(self, monkeypatch):
        import httpx
        from mcc_cli import _latest_pypi_version

        def fake_get(url, timeout=None):
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(httpx, "get", fake_get)
        assert _latest_pypi_version() is None


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


class TestHistoryEvents:
    @staticmethod
    def _history():
        from mcc_core import (
            Message,
            Role,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        return [
            Message.user("帮我修 bug"),
            Message(
                role=Role.assistant,
                content=[
                    TextBlock(text="先看文件"),
                    ToolUseBlock(id="t1", name="read_file", input={"path": "a.py"}),
                ],
            ),
            Message(
                role=Role.tool,
                content=[ToolResultBlock(tool_use_id="t1", content="x" * 600)],
            ),
            Message(role=Role.assistant, content=[TextBlock(text="修好了")]),
        ]

    def test_event_sequence(self):
        from mcc_cli import _history_events

        events = _history_events(self._history(), cost_usd=0.05)
        kinds = [type(e).__name__ for e in events]
        assert kinds == [
            "UserEcho",
            "TextDelta",
            "ToolCallStart",
            "ToolCallEnd",
            "TextDelta",
            "TurnEnd",
        ]
        assert events[-1].cost_usd == 0.05

    def test_tool_output_truncated_to_500(self):
        from mcc_cli import _history_events

        events = _history_events(self._history())
        end = [e for e in events if type(e).__name__ == "ToolCallEnd"][0]
        assert len(end.output) == 500

    def test_tool_error_flag_preserved(self):
        from mcc_cli import _history_events
        from mcc_core import Message, Role, ToolResultBlock

        history = [
            Message(
                role=Role.tool,
                content=[ToolResultBlock(tool_use_id="t9", content="boom", is_error=True)],
            ),
        ]
        events = _history_events(history)
        end = [e for e in events if type(e).__name__ == "ToolCallEnd"][0]
        assert end.is_error

    def test_empty_history_yields_only_turn_end(self):
        from mcc_cli import _history_events

        events = _history_events([])
        assert [type(e).__name__ for e in events] == ["TurnEnd"]
        assert events[0].cost_usd == 0.0

    def test_replay_renders_full_trace_in_app(self):
        from mcc_cli import _history_events
        from mcc_tui.app import App

        app = App(session_short_id="s")
        for ev in _history_events(self._history(), cost_usd=0.05):
            app.apply_event(ev)

        kinds = [i.kind for i in app.items]
        assert kinds == ["user", "assistant", "tool_call", "tool_result", "assistant"]
        assert app.items[2].call_id == "t1"
        assert app.items[2].status.value == "done"
        assert app.items[3].output == "x" * 500
        assert app.items[4].text == "修好了"
        assert app.cost_usd == 0.05
        assert app.streaming_text == ""
        assert not app.agent_busy


def _config(main="claude-opus-4-8", subagent=None, summarize=None):
    from mcc_config import Config, ModelConfig

    return Config(
        model=ModelConfig(main=main, subagent=subagent, summarize=summarize)
    )


class TestResolveModel:
    """模型解析统一走 mcc-config.resolve_model（config 对象注入）。"""

    def test_explicit_beats_env_and_json(self, monkeypatch):
        from mcc_config import resolve_model

        monkeypatch.setenv("MODEL", "env-model")
        cfg = _config(summarize="json-sum")
        assert resolve_model("summarize", "cli-model", config=cfg) == "cli-model"

    def test_env_beats_json(self, monkeypatch):
        from mcc_config import resolve_model

        monkeypatch.setenv("MODEL", "env-model")
        cfg = _config(main="json-main", summarize="json-sum")
        assert resolve_model("main", config=cfg) == "env-model"
        assert resolve_model("summarize", config=cfg) == "env-model"
        assert resolve_model("subagent", config=cfg) == "env-model"

    def test_json_kind_when_env_unset(self, monkeypatch):
        from mcc_config import resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        cfg = _config(main="json-main", summarize="json-sum")
        assert resolve_model("summarize", config=cfg) == "json-sum"
        assert resolve_model("main", config=cfg) == "json-main"

    def test_unconfigured_kind_falls_back_to_main(self, monkeypatch):
        from mcc_config import resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        cfg = _config(main="the-main")
        assert resolve_model("summarize", config=cfg) == "the-main"
        assert resolve_model("subagent", config=cfg) == "the-main"

    def test_default_when_nothing_configured(self, monkeypatch):
        from mcc_config import resolve_model

        monkeypatch.delenv("MODEL", raising=False)
        assert resolve_model("main", config=_config()) == "claude-opus-4-8"


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


class TestBuildSourceSkills:
    """skills 加载与 slash 命令接入 _build_source。"""

    @staticmethod
    def _write_skill(root: Path, name: str, desc: str = "d") -> None:
        d = root / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nbody",
            encoding="utf-8",
        )

    def test_catalog_injected_into_system(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        self._write_skill(tmp_path, "reviewer", "review code")
        source = _build_source(tmp_path, None, True, None)
        assert "Available Skills" in source.agent.system
        assert "reviewer" in source.agent.system

    def test_load_skill_registered(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        self._write_skill(tmp_path, "x")
        source = _build_source(tmp_path, None, True, None)
        assert source.agent.registry.get("load_skill") is not None

    def test_no_skills_flag_skips_loading(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        self._write_skill(tmp_path, "x")
        source = _build_source(tmp_path, None, True, None, no_skills=True)
        assert "Available Skills" not in source.agent.system
        assert source.agent.registry.get("load_skill") is None

    def test_empty_skills_dir_no_catalog(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(tmp_path, None, True, None)
        assert "Available Skills" not in source.agent.system
        assert source.agent.registry.get("load_skill") is None

    def test_slash_resolver_attached(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cmds = tmp_path / "commands"
        cmds.mkdir()
        (cmds / "test.md").write_text(
            "---\ncommand: /test\ndescription: d\n---\nrun pytest",
            encoding="utf-8",
        )
        source = _build_source(tmp_path, None, True, None)
        assert source._slash_resolver is not None
        assert source._slash_resolver("/test") == "run pytest"

    def test_slash_resolver_passes_through_unknown(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cmds = tmp_path / "commands"
        cmds.mkdir()
        (cmds / "test.md").write_text(
            "---\ncommand: /test\ndescription: d\n---\nbody", encoding="utf-8"
        )
        source = _build_source(tmp_path, None, True, None)
        assert source._slash_resolver("/unknown") is None

    def test_no_slash_resolver_when_disabled(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        cmds = tmp_path / "commands"
        cmds.mkdir()
        (cmds / "test.md").write_text(
            "---\ncommand: /test\ndescription: d\n---\nbody", encoding="utf-8"
        )
        source = _build_source(tmp_path, None, True, None, no_skills=True)
        assert source._slash_resolver is None

    def test_home_skills_loaded(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(home))
        self._write_skill(home / ".mcc", "global-skill", "global")
        source = _build_source(tmp_path, None, True, None)
        assert "global-skill" in source.agent.system
