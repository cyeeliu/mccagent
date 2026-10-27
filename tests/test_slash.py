"""slash 命令与首启向导测试：注册表 / 内置命令 / settings 写回 / 向导。"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcc_cli.slash import (
    SlashRegistry,
    SlashResult,
    build_default_registry,
)
from mcc_config import read_home_settings, update_home_setting
from mcc_config.models import Config
from mcc_core import Usage
from mcc_harness.goal import GoalController
from mcc_llm.mock import MockLlm, single_text_turn


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """隔离 HOME：settings 写回落到 tmp_path/.mcc/settings.json。"""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _source(**overrides: Any) -> SimpleNamespace:
    """轻量 AgentEventSource 替身：slash handler 只用到的属性。"""
    source = SimpleNamespace(
        agent=SimpleNamespace(model="claude-opus-4-8", history=[]),
        config=Config(),
        slash_registry=None,
        skills_catalog=None,
        session_short_id="abc12345",
        goal_controller=None,
    )
    for key, value in overrides.items():
        setattr(source, key, value)
    return source


# -- settings 写回 API --


class TestUpdateHomeSetting:
    def test_writes_nested_key(self, home: Path) -> None:
        update_home_setting(["model", "main"], "claude-x")
        data = json.loads((home / ".mcc" / "settings.json").read_text())
        assert data["model"]["main"] == "claude-x"

    def test_preserves_existing_content(self, home: Path) -> None:
        path = home / ".mcc" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": {"subagent": "haiku"}, "budget": {"max_usd_per_session": 9}}),
            encoding="utf-8",
        )
        update_home_setting(["model", "main"], "claude-y")
        data = read_home_settings()
        assert data["model"]["subagent"] == "haiku"
        assert data["budget"]["max_usd_per_session"] == 9
        assert data["model"]["main"] == "claude-y"

    def test_creates_intermediate_dicts(self, home: Path) -> None:
        update_home_setting(["reliability", "turn_timeout_secs"], 300.0)
        data = read_home_settings()
        assert data["reliability"]["turn_timeout_secs"] == 300.0

    def test_file_mode_is_600(self, home: Path) -> None:
        update_home_setting(["model", "main"], "x")
        mode = stat.S_IMODE((home / ".mcc" / "settings.json").stat().st_mode)
        assert mode == 0o600

    def test_empty_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="keys must not be empty"):
            update_home_setting([], "v")

    def test_read_missing_returns_empty(self, home: Path) -> None:
        assert read_home_settings() == {}

    def test_read_corrupt_returns_empty(self, home: Path) -> None:
        path = home / ".mcc" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json", encoding="utf-8")
        assert read_home_settings() == {}


# -- SlashRegistry --


class TestRegistry:
    def test_register_parse_dispatch(self) -> None:
        registry = SlashRegistry()
        seen: list[str] = []

        def handler(source: Any, args: str) -> SlashResult:
            seen.append(args)
            return SlashResult(text=f"got {args}")

        registry.register("ping", "ping test", handler)

        async def main() -> None:
            source = _source()
            result = await registry.dispatch("/ping hello world", source)
            assert result is not None
            assert result.text == "got hello world"
            assert seen == ["hello world"]

        import asyncio

        asyncio.run(main())

    def test_unknown_command_returns_none(self) -> None:
        registry = build_default_registry()
        import asyncio

        assert asyncio.run(registry.dispatch("/nonexistent", _source())) is None

    def test_non_slash_returns_none(self) -> None:
        registry = build_default_registry()
        import asyncio

        assert asyncio.run(registry.dispatch("hello", _source())) is None

    def test_async_handler(self) -> None:
        registry = SlashRegistry()

        async def handler(source: Any, args: str) -> SlashResult:
            return SlashResult(text="async ok")

        registry.register("async-cmd", "async", handler)

        import asyncio

        result = asyncio.run(registry.dispatch("/async-cmd", _source()))
        assert result is not None
        assert result.text == "async ok"

    def test_list_commands_sorted(self) -> None:
        registry = build_default_registry()
        names = [n for n, _ in registry.list_commands()]
        assert names == sorted(names)
        assert {"config", "goal", "help", "model", "status"} <= set(names)


# -- /help 与 /status --


class TestBuiltinHelpStatus:
    async def test_help_lists_commands_and_skills(self) -> None:
        registry = build_default_registry()
        source = _source(
            slash_registry=registry,
            skills_catalog="- my-skill: does things",
        )
        result = await registry.dispatch("/help", source)
        assert result is not None and result.text is not None
        assert "/model" in result.text
        assert "my-skill" in result.text

    async def test_help_without_skills(self) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/help", _source(slash_registry=registry))
        assert result is not None and result.text is not None
        assert "Skills" not in result.text

    async def test_status(self) -> None:
        registry = build_default_registry()
        source = _source(
            slash_registry=registry,
            agent=SimpleNamespace(model="claude-x", history=[1, 2]),
        )
        result = await registry.dispatch("/status", source)
        assert result is not None and result.text is not None
        assert "claude-x" in result.text
        assert "abc12345" in result.text
        assert "history messages: 2" in result.text


# -- /model --


class TestModelCommand:
    async def test_no_args_shows_current(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/model", source)
        assert result is not None and result.text is not None
        assert "claude-opus-4-8" in result.text

    async def test_switch_main_hot(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/model claude-haiku-4-5", source)
        assert result is not None and result.text is not None
        assert source.agent.model == "claude-haiku-4-5"
        assert "即刻生效" in result.text
        assert read_home_settings()["model"]["main"] == "claude-haiku-4-5"

    async def test_switch_with_kind_main(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        await registry.dispatch("/model main claude-x", source)
        assert source.agent.model == "claude-x"

    async def test_switch_subagent_writes_only(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/model subagent claude-haiku", source)
        assert result is not None and result.text is not None
        assert "下个会话生效" in result.text
        assert source.agent.model == "claude-opus-4-8"
        assert read_home_settings()["model"]["subagent"] == "claude-haiku"

    async def test_unknown_kind_rejected(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/model bogus x", _source())
        assert result is not None and result.text is not None
        assert "未知模型类型" in result.text


# -- /config --


class TestConfigCommand:
    async def test_summary(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/config", _source())
        assert result is not None and result.text is not None
        assert "model.main" in result.text
        assert "permissions.mode" in result.text

    async def test_get_key(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/config get budget.max_usd_per_session", _source())
        assert result is not None and result.text is not None
        assert "2.0" in result.text

    async def test_get_secret_masked(self, home: Path) -> None:
        update_home_setting(["llm", "api_key"], "sk-test-1234567890abcd")
        registry = build_default_registry()
        source = _source()
        from mcc_config import load as _load

        source.config = await _load(home)
        result = await registry.dispatch("/config get llm.api_key", source)
        assert result is not None and result.text is not None
        assert "sk-test-1234567890abcd" not in result.text
        assert "…abcd" in result.text

    async def test_get_unknown_key(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/config get bogus.key", _source())
        assert result is not None and result.text is not None
        assert "未知配置项" in result.text

    async def test_set_whitelisted(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch(
            "/config set budget.max_usd_per_session 5.5", source
        )
        assert result is not None and result.text is not None
        assert "5.5" in result.text
        assert read_home_settings()["budget"]["max_usd_per_session"] == 5.5
        assert source.config.budget.max_usd_per_session == 5.5

    async def test_set_model_main_hot(self, home: Path) -> None:
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        await registry.dispatch("/config set model.main claude-y", source)
        assert source.agent.model == "claude-y"
        assert source.config.model.main == "claude-y"

    async def test_set_rejected_key(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/config set hooks.pre_tool_use x", _source())
        assert result is not None and result.text is not None
        assert "不支持修改" in result.text

    async def test_set_invalid_type(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/config set budget.max_usd_per_session abc", _source()
        )
        assert result is not None and result.text is not None
        assert "不是合法" in result.text

    async def test_set_api_key_masked_in_reply(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/config set llm.api_key sk-secret-value-9999", _source()
        )
        assert result is not None and result.text is not None
        assert "sk-secret-value-9999" not in result.text
        assert read_home_settings()["llm"]["api_key"] == "sk-secret-value-9999"

    async def test_usage_on_bad_subcommand(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/config bogus", _source())
        assert result is not None and result.text is not None
        assert "用法" in result.text


# -- /goal --


class TestGoalCommand:
    async def test_goal_set_replaces_prompt(self, home: Path) -> None:
        registry = build_default_registry()
        controller = GoalController(evaluator=None)
        source = _source(slash_registry=registry, goal_controller=controller)
        result = await registry.dispatch("/goal make tests pass", source)
        assert result is not None
        assert result.prompt == "make tests pass"
        assert "goal set" in (result.text or "")
        assert controller.status() != "no goal"

    async def test_goal_clear(self, home: Path) -> None:
        registry = build_default_registry()
        controller = GoalController(evaluator=None)
        controller.set_goal("x")
        source = _source(slash_registry=registry, goal_controller=controller)
        result = await registry.dispatch("/goal clear", source)
        assert result is not None
        assert result.prompt is None
        assert "cleared" in (result.text or "")

    async def test_goal_status(self, home: Path) -> None:
        registry = build_default_registry()
        controller = GoalController(evaluator=None)
        source = _source(slash_registry=registry, goal_controller=controller)
        result = await registry.dispatch("/goal", source)
        assert result is not None and result.text is not None
        assert "goal" in result.text

    async def test_goal_disabled(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/goal x", _source(slash_registry=registry))
        assert result is not None and result.text is not None
        assert "未启用" in result.text


# -- AgentEventSource 集成 --


class TestEventSourceIntegration:
    def _build(self, registry=None, goal=None):
        from mcc_cli.events import AgentEventSource

        llm = MockLlm([single_text_turn("agent reply")])
        import tempfile

        from mcc_core import ToolContext
        from mcc_harness import AgentLoopBuilder
        from mcc_tools import ToolRegistry

        ctx = ToolContext(cwd=Path(tempfile.mkdtemp()), session_id="t", depth=0)
        agent = (
            AgentLoopBuilder(llm, ToolRegistry(), ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )
        return AgentEventSource(
            agent,
            slash_registry=registry,
            goal_controller=goal,
            config=Config(),
        ), llm

    async def test_local_command_skips_agent_loop(self) -> None:
        source, llm = self._build(registry=build_default_registry())
        events = [e async for e in source.run("/help")]
        assert llm.call_count == 0
        texts = [e.text for e in events if hasattr(e, "text")]
        assert any("/model" in t for t in texts)

    async def test_goal_set_prompt_continues_to_agent(self) -> None:
        controller = GoalController(evaluator=None)
        source, llm = self._build(
            registry=build_default_registry(), goal=controller
        )
        events = [e async for e in source.run("/goal finish the task")]
        assert llm.call_count == 1
        texts = [e.text for e in events if hasattr(e, "text")]
        assert any("agent reply" in t for t in texts)

    async def test_unknown_slash_falls_to_agent(self) -> None:
        source, llm = self._build(registry=build_default_registry())
        _ = [e async for e in source.run("/not-a-command")]
        assert llm.call_count == 1

    async def test_no_registry_passthrough(self) -> None:
        source, llm = self._build(registry=None)
        _ = [e async for e in source.run("/help")]
        assert llm.call_count == 1


# -- 首启向导 --


class TestOnboardingWizard:
    def _run(self, home: Path, answers: list[str]) -> bool:
        import click
        from mcc_cli import _run_onboarding_wizard

        queue = list(answers)

        def fake_prompt(*args: Any, **kwargs: Any) -> str:
            if not queue:
                raise click.Abort()
            return queue.pop(0)

        import mcc_cli

        original = mcc_cli.click.prompt
        mcc_cli.click.prompt = fake_prompt
        try:
            return _run_onboarding_wizard()
        finally:
            mcc_cli.click.prompt = original

    def test_anthropic_path(self, home: Path) -> None:
        ok = self._run(home, ["anthropic", "sk-ant-testkey", "claude-opus-4-8"])
        assert ok is True
        data = read_home_settings()
        assert data["llm"]["api_key"] == "sk-ant-testkey"
        assert data["model"]["main"] == "claude-opus-4-8"
        assert "base_url" not in data.get("llm", {})

    def test_gateway_path(self, home: Path) -> None:
        ok = self._run(
            home,
            ["gateway", "http://gw:8000/v1", "", "deepseek-chat"],
        )
        assert ok is True
        data = read_home_settings()
        assert data["llm"]["base_url"] == "http://gw:8000/v1"
        assert "api_key" not in data["llm"]
        assert data["model"]["main"] == "deepseek-chat"

    def test_abort_returns_false(self, home: Path) -> None:
        ok = self._run(home, [])
        assert ok is False
        assert read_home_settings() == {}
