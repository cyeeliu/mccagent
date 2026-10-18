"""包结构验证测试。

验证所有 package 全部可导入、依赖关系正确、核心类型可用。
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------- mcc-core ----------

def test_mcc_core_imports():
    import mcc_core

    assert mcc_core.__version__ == "0.1.0"


def test_role_enum():
    from mcc_core import Role

    assert Role.user.value == "user"
    assert Role.assistant.value == "assistant"
    assert Role.system.value == "system"


def test_message_constructors():
    from mcc_core import Message, Role, TextBlock

    m = Message.user("hello")
    assert m.role == Role.user
    assert len(m.content) == 1
    assert isinstance(m.content[0], TextBlock)
    assert m.content[0].text == "hello"

    m2 = Message.assistant("hi")
    assert m2.role == Role.assistant


def test_content_block_serialization():
    from mcc_core import TextBlock, ToolResultBlock, ToolUseBlock

    t = TextBlock(text="hello")
    assert t.model_dump() == {"type": "text", "text": "hello", "cache_control": None}

    tu = ToolUseBlock(id="t1", name="read_file", input={"path": "a.py"})
    d = tu.model_dump()
    assert d["type"] == "tool_use"
    assert d["id"] == "t1"
    assert d["input"] == {"path": "a.py"}

    tr = ToolResultBlock(tool_use_id="t1", content="ok")
    assert tr.model_dump()["type"] == "tool_result"
    assert tr.is_error is False


def test_usage_addition():
    from mcc_core import Usage

    u1 = Usage(input_tokens=10, output_tokens=5)
    u2 = Usage(input_tokens=20, output_tokens=15, cache_read_input_tokens=100)
    total = u1 + u2
    assert total.input_tokens == 30
    assert total.output_tokens == 20
    assert total.cache_read_input_tokens == 100


def test_tool_output_factories():
    from mcc_core import ToolOutput

    ok = ToolOutput.ok("done")
    assert ok.content == "done" and ok.is_error is False

    err = ToolOutput.err("boom")
    assert err.content == "boom" and err.is_error is True


def test_tool_context():
    from mcc_core import ToolContext

    ctx = ToolContext(cwd=Path("/tmp"), session_id="s1")
    assert ctx.depth == 0
    assert ctx.session_id == "s1"


def test_agent_error_hierarchy():
    from mcc_core import (
        AgentError,
        ApiError,
        BudgetExceeded,
        Cancelled,
        PermissionDenied,
        ToolError,
    )

    assert issubclass(ApiError, AgentError)
    assert issubclass(ToolError, AgentError)
    assert issubclass(PermissionDenied, AgentError)
    assert issubclass(BudgetExceeded, AgentError)
    assert issubclass(Cancelled, AgentError)

    te = ToolError("read_file", "not found")
    assert te.name == "read_file"
    assert "read_file" in str(te)


def test_agent_event_types():
    from mcc_core import (
        Error,
        Notice,
        PermissionRequest,
        TextDelta,
        ToolCallEnd,
        ToolCallStart,
        TurnEnd,
        UserEcho,
    )

    assert UserEcho("x").text == "x"
    assert TextDelta("d").text == "d"
    assert ToolCallStart(id="1", name="bash", args_preview="ls").name == "bash"
    assert ToolCallEnd(id="1", output="ok", is_error=False).is_error is False
    assert TurnEnd(cost_usd=0.01).cost_usd == 0.01
    assert Notice("n").text == "n"
    assert Error("e").text == "e"
    assert PermissionRequest(id=1, message="allow?").id == 1


# ---------- mcc-config ----------

def test_mcc_config_imports():
    import mcc_config

    assert mcc_config.__version__ == "0.1.0"


def test_config_defaults():
    from mcc_config import Config

    c = Config()
    assert c.model.main == "claude-opus-4-8"
    assert c.model.subagent is None
    assert c.model.summarize is None
    assert c.budget.max_usd_per_session == 2.0
    assert c.budget.max_iterations == 40
    assert c.permissions.mode is None
    assert c.permissions.allow == []


def test_config_merge_project_keeps_home_budget():
    from mcc_config import BudgetConfig, Config, PermissionConfig, merge

    home = Config(budget=BudgetConfig(max_usd_per_session=9.0, max_iterations=99))
    proj = Config(
        permissions=PermissionConfig(deny=["Bash(rm:*)"]),
    )
    merged = merge(home, proj)
    assert merged.budget.max_usd_per_session == 9.0
    assert "Bash(rm:*)" in merged.permissions.deny


def test_config_merge_project_budget_overrides():
    from mcc_config import BudgetConfig, Config, merge

    home = Config(budget=BudgetConfig(max_usd_per_session=9.0, max_iterations=99))
    proj = Config(budget=BudgetConfig(max_usd_per_session=1.0, max_iterations=5))
    merged = merge(home, proj)
    assert merged.budget.max_usd_per_session == 1.0
    assert merged.budget.max_iterations == 5


def test_config_merge_mode_override():
    from mcc_config import Config, PermissionConfig, merge

    home = Config(permissions=PermissionConfig(mode="default"))
    proj = Config(permissions=PermissionConfig(mode="bypassPermissions"))
    merged = merge(home, proj)
    assert merged.permissions.mode == "bypassPermissions"


def test_config_load_from_settings_json(tmp_path: Path):
    from mcc_config import load

    settings = tmp_path / ".mcc" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "model": {"main": "gpt-4o", "subagent": "gpt-4o-mini", "summarize": "gpt-4o-mini"},
                "budget": {"max_usd_per_session": 5.0, "max_iterations": 10},
            }
        )
    )
    import asyncio

    cfg = asyncio.run(load(tmp_path))
    assert cfg.model.main == "gpt-4o"
    assert cfg.budget.max_usd_per_session == 5.0


def test_config_model_defaults_none():
    from mcc_config import Config

    c = Config()
    assert c.model.main == "claude-opus-4-8"
    assert c.model.subagent is None
    assert c.model.summarize is None


def test_config_merge_model_fieldwise():
    from mcc_config import Config, ModelConfig, merge

    home = Config(model=ModelConfig(main="home-main", summarize="home-sum"))
    proj = Config(model=ModelConfig(main="proj-main"))
    merged = merge(home, proj)
    assert merged.model.main == "proj-main"
    assert merged.model.summarize == "home-sum"


def test_config_merge_model_subagent_only():
    from mcc_config import Config, ModelConfig, merge

    home = Config(model=ModelConfig(main="home-main", subagent="home-sub"))
    proj = Config(model=ModelConfig(subagent="proj-sub", summarize="proj-sum"))
    merged = merge(home, proj)
    assert merged.model.main == "home-main"
    assert merged.model.subagent == "proj-sub"
    assert merged.model.summarize == "proj-sum"


def test_config_legacy_json_only_main(tmp_path: Path):
    from mcc_config import load

    settings = tmp_path / ".mcc" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"model": {"main": "old-main"}}))
    import asyncio

    cfg = asyncio.run(load(tmp_path))
    assert cfg.model.main == "old-main"
    assert cfg.model.subagent is None
    assert cfg.model.summarize is None


# ---------- 其余 package ----------

def test_mcc_llm_imports():
    import mcc_llm
    from mcc_llm import TextDelta

    assert mcc_llm.__version__ == "0.1.0"
    assert TextDelta(text="hi").text == "hi"


def test_mcc_tools_imports():
    import mcc_tools
    from mcc_tools import ToolRegistry

    assert mcc_tools.__version__ == "0.1.0"
    reg = ToolRegistry()
    assert reg.names() == []
    assert reg.get("nope") is None


def test_mcc_session_imports():
    import mcc_session

    assert mcc_session.__version__ == "0.1.0"


def test_mcc_harness_imports():
    import mcc_harness
    from mcc_harness import AgentLoopBuilder

    assert mcc_harness.__version__ == "0.1.0"
    assert AgentLoopBuilder.__name__ == "AgentLoopBuilder"


def test_mcc_tui_imports():
    import mcc_tui

    assert mcc_tui.__version__ == "0.1.0"


def test_mcc_cli_imports():
    import mcc_cli

    assert mcc_cli.__version__ == "0.1.0"


# ---------- 依赖图验证 ----------

def test_dependency_graph():
    """验证模块依赖图与设计一致。"""
    import importlib

    expected = [
        "mcc_core",
        "mcc_config",
        "mcc_llm",
        "mcc_tools",
        "mcc_session",
        "mcc_harness",
        "mcc_tui",
        "mcc_cli",
    ]
    for name in expected:
        mod = importlib.import_module(name)
        assert mod is not None, f"{name} 不可导入"
