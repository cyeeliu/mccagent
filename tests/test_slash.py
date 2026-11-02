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

    @pytest.mark.parametrize(
        "cmd, expected",
        [("/model claude-x", "claude-x"), ("/model main claude-y", "claude-y")],
    )
    async def test_main_switch_sets_model_update(
        self, home: Path, cmd: str, expected: str
    ) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(cmd, _source(slash_registry=registry))
        assert result is not None
        assert result.model_update == expected

    async def test_subagent_switch_no_model_update(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/model subagent claude-h", _source(slash_registry=registry)
        )
        assert result is not None
        assert result.model_update is None


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

    async def test_set_model_main_sets_model_update(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/config set model.main claude-z", _source(slash_registry=registry)
        )
        assert result is not None
        assert result.model_update == "claude-z"

    async def test_set_model_subagent_no_model_update(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/config set model.subagent claude-s", _source(slash_registry=registry)
        )
        assert result is not None
        assert result.model_update is None

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

    @pytest.mark.parametrize("cmd", ["/help", "/status", "/config", "/model"])
    async def test_display_command_emits_turn_end(self, cmd: str) -> None:
        """纯展示型 slash 命令发零成本 TurnEnd，使 TUI 复位 agent_busy。"""
        from mcc_core import TurnEnd

        source, _ = self._build(registry=build_default_registry())
        events = [e async for e in source.run(cmd)]
        turn_ends = [e for e in events if isinstance(e, TurnEnd)]
        assert len(turn_ends) == 1
        assert turn_ends[0].cost_usd == 0.0

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            ("/model claude-new", "claude-new"),
            ("/config set model.main claude-new2", "claude-new2"),
        ],
    )
    async def test_model_switch_emits_model_changed(
        self, home: Path, cmd: str, expected: str
    ) -> None:
        """切主模型经事件流通知前端：/model 与 /config set model.main 两入口。"""
        from mcc_core import ModelChanged

        source, _ = self._build(registry=build_default_registry())
        events = [e async for e in source.run(cmd)]
        changed = [e for e in events if isinstance(e, ModelChanged)]
        assert len(changed) == 1
        assert changed[0].model_id == expected

    async def test_subagent_switch_emits_no_model_changed(self, home: Path) -> None:
        from mcc_core import ModelChanged

        source, _ = self._build(registry=build_default_registry())
        events = [e async for e in source.run("/model subagent claude-s")]
        assert not any(isinstance(e, ModelChanged) for e in events)

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


# -- skills 补全 --


class TestSkillsCompletion:
    """skills 名注入补全 provider：输入 / 时 skill 候选出现在浮层。"""

    def test_skill_names_appear_in_completions(self) -> None:
        from mcc_tui.completions import make_provider

        provider = make_provider(
            [("help", "帮助"), ("status", "状态")],
            skill_items=[("code-review", "审查代码"), ("debug-fix", "修复缺陷")],
        )
        inserts = [item.insert for item in provider("/")]
        assert "/help" in inserts
        assert "/status" in inserts
        assert "/code-review" in inserts
        assert "/debug-fix" in inserts

    @pytest.mark.parametrize(
        "prefix, expected",
        [
            ("/", ["/code-review", "/debug-fix"]),
            ("/code", ["/code-review"]),
            ("/debug", ["/debug-fix"]),
            ("/xyz", []),
        ],
    )
    def test_skill_prefix_filter(self, prefix: str, expected: list[str]) -> None:
        from mcc_tui.completions import make_provider

        provider = make_provider(
            [], skill_items=[("code-review", ""), ("debug-fix", "")]
        )
        inserts = [item.insert for item in provider(prefix)]
        assert inserts == expected

    def test_no_skill_names_no_skill_items(self) -> None:
        from mcc_tui.completions import make_provider

        provider = make_provider([("help", "帮助")])
        inserts = [item.insert for item in provider("/")]
        assert inserts == ["/help"]

    def test_skill_hint_uses_description(self) -> None:
        """skill 候选的 hint 为 manifest.description 而非硬编码 'skill'。"""
        from mcc_tui.completions import make_provider

        provider = make_provider([], skill_items=[("my-skill", "修复代码缺陷")])
        items = provider("/")
        assert len(items) == 1
        assert items[0].hint == "修复代码缺陷"
        assert items[0].label == "/my-skill"

    def test_skill_hint_fallback_when_no_description(self) -> None:
        """description 为空时 hint 回退为 'skill'。"""
        from mcc_tui.completions import make_provider

        provider = make_provider([], skill_items=[("my-skill", "")])
        items = provider("/")
        assert items[0].hint == "skill"


# -- /resume --


def _write_session(home: Path, sid: str, turns: int = 1, title: str | None = None) -> None:
    """在隔离 HOME 下直接落盘一个会话的 meta + JSONL 文件。"""
    from datetime import UTC, datetime

    from mcc_core import Message, Role, TextBlock
    from mcc_session.models import SessionMeta, TurnSnapshot

    base = home / ".mcc" / "sessions"
    day = base / datetime.now(UTC).strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)

    meta = SessionMeta(
        id=sid,
        started_at=datetime.now(UTC),
        cwd=str(home),
        turns=turns,
        cost_usd=0.001 * turns,
        status="completed",
        title=title,
    )
    (day / f"{sid}.meta.json").write_text(
        meta.model_dump_json(indent=2), encoding="utf-8"
    )

    lines: list[str] = []
    for i in range(turns):
        snapshot = TurnSnapshot(
            ts=datetime.now(UTC),
            iteration=i,
            request_messages=[
                Message(role=Role.user, content=[TextBlock(text=f"turn {i}")])
            ],
            assistant_blocks=[TextBlock(text=f"reply {i}")],
            usage=Usage(),
            model="test-model",
            cost_usd=0.001,
        )
        lines.append(snapshot.model_dump_json())
    (day / f"{sid}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestResumeCommand:
    """/resume：列出会话与就地恢复历史。"""

    async def test_list_no_sessions(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch("/resume", _source(slash_registry=registry))
        assert result is not None and result.text is not None
        assert "没有历史会话" in result.text

    async def test_list_returns_select_structure(self, home: Path) -> None:
        """无参 /resume 返回 select 候选供 TUI 弹出选择浮层。"""
        _write_session(home, "abc12345", turns=2, title="my task")
        _write_session(home, "xyz99999", turns=1)
        registry = build_default_registry()
        result = await registry.dispatch("/resume", _source(slash_registry=registry))
        assert result is not None
        assert result.select is not None
        assert result.select_title.startswith("选择要恢复的会话")
        assert result.select_command == "resume"
        assert len(result.select) == 2
        labels = [label for label, _value in result.select]
        values = [_value for _label, _value in result.select]
        assert "abc12345" in values
        assert "xyz99999" in values
        assert any("my task" in label for label in labels)
        assert any("2turn" in label for label in labels)

    async def test_resume_by_id_loads_history(self, home: Path) -> None:
        _write_session(home, "xyz99999", turns=1)
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/resume xyz99999", source)
        assert result is not None and result.text is not None
        assert "已恢复" in result.text
        assert "xyz99999" in result.text
        assert len(source.agent.history) > 0
        assert source.session_short_id == "xyz99999"

    async def test_resume_by_id_preserves_multi_turn_history(self, home: Path) -> None:
        _write_session(home, "multi001", turns=3)
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/resume multi001", source)
        assert result is not None and result.text is not None
        assert "3 turn" in result.text
        assert len(source.agent.history) >= 2

    async def test_resume_unknown_id(self, home: Path) -> None:
        registry = build_default_registry()
        result = await registry.dispatch(
            "/resume nosuchid", _source(slash_registry=registry)
        )
        assert result is not None and result.text is not None
        assert "错误" in result.text

    async def test_resume_registered_in_default_registry(self) -> None:
        registry = build_default_registry()
        commands = dict(registry.list_commands())
        assert "resume" in commands

    async def test_list_marks_current_session(self, home: Path) -> None:
        """无参 /resume 在 label 中用 ● 标记当前会话并排到首位。"""
        _write_session(home, "abc12345", turns=2, title="current")
        _write_session(home, "xyz99999", turns=1, title="other")
        registry = build_default_registry()
        source = _source(slash_registry=registry, session_short_id="abc12345")
        result = await registry.dispatch("/resume", source)
        assert result is not None and result.select is not None
        labels = [label for label, _ in result.select]
        assert any(label.startswith("●") and "current" in label for label in labels)
        assert any(label.startswith("  ") and "other" in label for label in labels)
        # 当前会话应排在第一位
        assert "current" in labels[0]

    async def test_resume_by_id_generates_replay(self, home: Path) -> None:
        """/resume <id> 在 SlashResult.replay 中携带历史重放事件。"""
        from mcc_core import TextDelta, TurnEnd, UserEcho

        _write_session(home, "xyz99999", turns=1)
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/resume xyz99999", source)
        assert result is not None
        assert result.replay is not None
        assert len(result.replay) > 0
        assert any(isinstance(e, UserEcho) for e in result.replay)
        assert any(isinstance(e, TextDelta) for e in result.replay)
        assert isinstance(result.replay[-1], TurnEnd)

    async def test_resume_by_id_sets_session_update(self, home: Path) -> None:
        """/resume <id> 在 SlashResult.session_update 中携带新会话 id。"""
        _write_session(home, "xyz99999", turns=1)
        registry = build_default_registry()
        source = _source(slash_registry=registry)
        result = await registry.dispatch("/resume xyz99999", source)
        assert result is not None
        assert result.session_update == "xyz99999"


class TestEventSourceSkillItems:
    """AgentEventSource 暴露 skill_items 供 TUI 补全消费。"""

    def test_skill_items_defaults_to_empty_list(self) -> None:
        from mcc_cli.events import AgentEventSource

        source = AgentEventSource(agent=SimpleNamespace())
        assert source.skill_items == []

    def test_skill_items_stored(self) -> None:
        from mcc_cli.events import AgentEventSource

        source = AgentEventSource(
            agent=SimpleNamespace(),
            skill_items=[("code-review", "审查代码"), ("debug-fix", "修复缺陷")],
        )
        assert source.skill_items == [
            ("code-review", "审查代码"),
            ("debug-fix", "修复缺陷"),
        ]


# -- /resume 选择浮层事件流 --


class TestResumeSelectionFlow:
    """/resume 无参经事件流：SelectionRequest → 用户选择 → 重新 dispatch。"""

    def _build(self):
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
            slash_registry=build_default_registry(),
            config=Config(),
        )

    async def test_headless_fallback_lists_text(self, home: Path) -> None:
        """selection_responses 为 None 时回退为文本列出候选。"""
        from mcc_core import Notice, TurnEnd

        _write_session(home, "abc12345", turns=2, title="my task")
        source = self._build()
        events = [e async for e in source.run("/resume")]
        notices = [e for e in events if isinstance(e, Notice)]
        assert any("abc12345" in e.text for e in notices)
        assert any(isinstance(e, TurnEnd) for e in events)

    async def test_selection_request_then_load(self, home: Path) -> None:
        """TUI 模式：发 SelectionRequest → 用户 Enter → 重新 dispatch 加载历史。"""
        import asyncio

        from mcc_core import Notice, SelectionDecision, SelectionRequest, TurnEnd

        _write_session(home, "abc12345", turns=1)
        source = self._build()
        sel_queue: asyncio.Queue = asyncio.Queue()
        source.selection_responses = sel_queue
        events = []
        async for ev in source.run("/resume"):
            events.append(ev)
            if isinstance(ev, SelectionRequest):
                sel_queue.put_nowait(
                    SelectionDecision(request_id=ev.id, value="abc12345")
                )
        assert any(isinstance(e, SelectionRequest) for e in events)
        notices = [e for e in events if isinstance(e, Notice)]
        assert any("已恢复" in e.text for e in notices)
        assert any(isinstance(e, TurnEnd) for e in events)

    async def test_selection_cancel_ends_turn(self, home: Path) -> None:
        """用户 Esc 取消选择 → 直接 TurnEnd，不加载任何会话。"""
        import asyncio

        from mcc_core import SelectionDecision, SelectionRequest, TurnEnd

        _write_session(home, "abc12345", turns=1)
        source = self._build()
        sel_queue: asyncio.Queue = asyncio.Queue()
        source.selection_responses = sel_queue
        events = []
        async for ev in source.run("/resume"):
            events.append(ev)
            if isinstance(ev, SelectionRequest):
                sel_queue.put_nowait(
                    SelectionDecision(request_id=ev.id, value=None)
                )
        assert any(isinstance(e, SelectionRequest) for e in events)
        assert any(isinstance(e, TurnEnd) for e in events)
        from mcc_core import Notice

        notices = [e for e in events if isinstance(e, Notice)]
        assert not any("已恢复" in e.text for e in notices)

    async def test_selection_yields_replay_and_session_changed(self, home: Path) -> None:
        """TUI 模式：用户选中 → 事件流包含 replay 事件与 SessionChanged。"""
        import asyncio

        from mcc_core import (
            SelectionDecision,
            SelectionRequest,
            SessionChanged,
            TextDelta,
            UserEcho,
        )

        _write_session(home, "abc12345", turns=1)
        source = self._build()
        sel_queue: asyncio.Queue = asyncio.Queue()
        source.selection_responses = sel_queue
        events = []
        async for ev in source.run("/resume"):
            events.append(ev)
            if isinstance(ev, SelectionRequest):
                sel_queue.put_nowait(
                    SelectionDecision(request_id=ev.id, value="abc12345")
                )
        assert any(isinstance(e, SessionChanged) and e.session_id == "abc12345"
                   for e in events)
        assert any(isinstance(e, UserEcho) for e in events)
        assert any(isinstance(e, TextDelta) for e in events)
        assert source.session_short_id == "abc12345"
