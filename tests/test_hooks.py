"""Hooks 机制测试：pre-tool / post-tool / lifecycle 钩子。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from mcc_core import ToolContext, Usage
from mcc_core.event import Notice, PermissionRequest, ToolCallEnd, TurnEnd
from mcc_harness import (
    AgentLoopBuilder,
    HookContext,
    HookRegistry,
    HookResult,
    HookType,
    PermissionChecker,
    PythonHook,
    ShellHook,
)
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="hooks-test")


@pytest.fixture
def registry():
    return default_registry()


# ---------- HookRegistry ----------


@pytest.mark.asyncio
async def test_hook_registry_register_and_get():
    reg = HookRegistry()

    async def cb(ctx: HookContext) -> HookResult:
        return HookResult()

    reg.add_python(HookType.PRE_TOOL, cb)
    assert len(reg.get(HookType.PRE_TOOL)) == 1
    assert len(reg.get(HookType.POST_TOOL)) == 0


@pytest.mark.asyncio
async def test_hook_registry_run_hooks_empty():
    reg = HookRegistry()
    ctx = HookContext(hook_type=HookType.PRE_TOOL)
    results = await reg.run_hooks(HookType.PRE_TOOL, ctx)
    assert results == []


@pytest.mark.asyncio
async def test_hook_registry_run_hooks_multiple():
    reg = HookRegistry()
    call_order: list[str] = []

    async def cb1(ctx: HookContext) -> HookResult:
        call_order.append("cb1")
        return HookResult(message="from cb1")

    async def cb2(ctx: HookContext) -> HookResult:
        call_order.append("cb2")
        return HookResult(message="from cb2")

    reg.add_python(HookType.PRE_TOOL, cb1)
    reg.add_python(HookType.PRE_TOOL, cb2)

    results = await reg.run_hooks(HookType.PRE_TOOL, HookContext(hook_type=HookType.PRE_TOOL))
    assert call_order == ["cb1", "cb2"]
    assert len(results) == 2
    assert results[0].message == "from cb1"
    assert results[1].message == "from cb2"


@pytest.mark.asyncio
async def test_hook_registry_exception_does_not_propagate():
    reg = HookRegistry()

    async def bad_cb(ctx: HookContext) -> HookResult:
        raise RuntimeError("boom")

    reg.add_python(HookType.PRE_TOOL, bad_cb)
    results = await reg.run_hooks(HookType.PRE_TOOL, HookContext(hook_type=HookType.PRE_TOOL))
    assert len(results) == 1
    assert "hook error" in results[0].message


# ---------- PythonHook ----------


@pytest.mark.asyncio
async def test_python_hook_basic():
    async def cb(ctx: HookContext) -> HookResult:
        return HookResult(allow=False, message="denied by test")

    hook = PythonHook(HookType.PRE_TOOL, cb)
    assert hook.hook_type == HookType.PRE_TOOL
    result = await hook.run(HookContext(hook_type=HookType.PRE_TOOL))
    assert result.allow is False
    assert result.message == "denied by test"


@pytest.mark.asyncio
async def test_python_hook_modify_input():
    async def cb(ctx: HookContext) -> HookResult:
        return HookResult(modified_input={"path": "modified.txt"})

    hook = PythonHook(HookType.PRE_TOOL, cb)
    result = await hook.run(
        HookContext(hook_type=HookType.PRE_TOOL, tool_input={"path": "original.txt"})
    )
    assert result.modified_input == {"path": "modified.txt"}


# ---------- ShellHook ----------


@pytest.mark.asyncio
async def test_shell_hook_allow(tmp_path: Path):
    script = tmp_path / "allow.sh"
    script.write_text("#!/usr/bin/env bash\necho 'allowed'\nexit 0\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script)
    result = await hook.run(
        HookContext(hook_type=HookType.PRE_TOOL, tool_name="read_file", cwd=tmp_path)
    )
    assert result.allow is True
    assert result.message == "allowed"


@pytest.mark.asyncio
async def test_shell_hook_deny(tmp_path: Path):
    script = tmp_path / "deny.sh"
    script.write_text("#!/usr/bin/env bash\necho 'forbidden' >&2\nexit 1\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script)
    result = await hook.run(
        HookContext(hook_type=HookType.PRE_TOOL, tool_name="run_bash", cwd=tmp_path)
    )
    assert result.allow is False
    assert "forbidden" in result.message


@pytest.mark.asyncio
async def test_shell_hook_env_vars(tmp_path: Path):
    script = tmp_path / "check_env.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'echo "tool=$MCC_TOOL_NAME type=$MCC_HOOK_TYPE"\n'
        "exit 0\n"
    )
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.POST_TOOL, script)
    result = await hook.run(
        HookContext(
            hook_type=HookType.POST_TOOL,
            tool_name="write_file",
            cwd=tmp_path,
        )
    )
    assert "tool=write_file" in result.message
    assert "type=post_tool" in result.message


@pytest.mark.asyncio
async def test_shell_hook_timeout(tmp_path: Path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/usr/bin/env bash\nsleep 5\nexit 0\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script, timeout=0.1)
    result = await hook.run(
        HookContext(hook_type=HookType.PRE_TOOL, cwd=tmp_path)
    )
    assert "hook error" in result.message


@pytest.mark.asyncio
async def test_shell_hook_not_found():
    hook = ShellHook(HookType.PRE_TOOL, "/nonexistent/script.sh")
    result = await hook.run(HookContext(hook_type=HookType.PRE_TOOL))
    assert "hook error" in result.message


# ---------- Pre-tool hooks in AgentLoop ----------


def _tool_turn(tool_id: str, tool_name: str, tool_input: dict):
    return [
        ToolUseStart(id=tool_id, name=tool_name),
        ToolUseInputDelta(partial_json=json.dumps(tool_input)),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


@pytest.mark.asyncio
async def test_pre_tool_hook_blocks(ctx, registry, tmp_path):
    async def block_hook(hctx: HookContext) -> HookResult:
        if hctx.tool_name == "run_bash":
            return HookResult(allow=False, message="bash blocked by hook")
        return HookResult()

    hooks = HookRegistry()
    hooks.add_python(HookType.PRE_TOOL, block_hook)

    llm = MockLlm([
        _tool_turn("t1", "run_bash", {"command": "echo hi"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    events = []
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    while not queue.empty():
        events.append(queue.get_nowait())

    tool_end = [e for e in events if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is True
    assert "blocked by hook" in tool_end.output


@pytest.mark.asyncio
async def test_pre_tool_hook_modifies_input(ctx, registry, tmp_path):
    async def modify_hook(hctx: HookContext) -> HookResult:
        if hctx.tool_name == "write_file":
            return HookResult(modified_input={"path": "redirected.txt", "content": "hooked"})
        return HookResult()

    hooks = HookRegistry()
    hooks.add_python(HookType.PRE_TOOL, modify_hook)

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "original.txt", "content": "original"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)

    assert (tmp_path / "redirected.txt").read_text() == "hooked"
    assert not (tmp_path / "original.txt").exists()


# ---------- Post-tool hooks ----------


@pytest.mark.asyncio
async def test_post_tool_hook_modifies_output(ctx, registry):
    async def modify_output_hook(hctx: HookContext) -> HookResult:
        if hctx.tool_name == "read_file":
            return HookResult(modified_output="OVERRIDDEN OUTPUT")
        return HookResult()

    hooks = HookRegistry()
    hooks.add_python(HookType.POST_TOOL, modify_output_hook)

    test_file = ctx.cwd / "test.txt"
    test_file.write_text("original content")

    llm = MockLlm([
        _tool_turn("t1", "read_file", {"path": "test.txt"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    events = []
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    while not queue.empty():
        events.append(queue.get_nowait())

    tool_end = [e for e in events if isinstance(e, ToolCallEnd)][0]
    assert "OVERRIDDEN" in tool_end.output


# ---------- Lifecycle hooks ----------


@pytest.mark.asyncio
async def test_on_turn_start_hook(ctx, registry):
    messages_seen: list[str] = []

    async def turn_start_hook(hctx: HookContext) -> HookResult:
        messages_seen.append(f"turn {hctx.iteration}")
        return HookResult(message=f"starting turn {hctx.iteration}")

    hooks = HookRegistry()
    hooks.add_python(HookType.ON_TURN_START, turn_start_hook)

    llm = MockLlm([single_text_turn("hello")])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    events = []
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    while not queue.empty():
        events.append(queue.get_nowait())

    assert messages_seen == ["turn 1"]
    notices = [e for e in events if isinstance(e, Notice)]
    assert any("starting turn 1" in n.text for n in notices)


@pytest.mark.asyncio
async def test_on_turn_end_hook(ctx, registry):
    async def turn_end_hook(hctx: HookContext) -> HookResult:
        return HookResult(message=f"turn ended, cost=${hctx.cost_usd:.6f}")

    hooks = HookRegistry()
    hooks.add_python(HookType.ON_TURN_END, turn_end_hook)

    llm = MockLlm([single_text_turn("hello")])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    events = []
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    while not queue.empty():
        events.append(queue.get_nowait())

    notices = [e for e in events if isinstance(e, Notice)]
    assert any("turn ended" in n.text for n in notices)


@pytest.mark.asyncio
async def test_lifecycle_hooks_multi_turn(ctx, registry, tmp_path):
    turn_starts: list[int] = []

    async def turn_start_hook(hctx: HookContext) -> HookResult:
        turn_starts.append(hctx.iteration)
        return HookResult()

    hooks = HookRegistry()
    hooks.add_python(HookType.ON_TURN_START, turn_start_hook)

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "a.txt", "content": "A"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)

    assert turn_starts == [1, 2]


# ---------- Interactive permission (Ask → PermissionRequest event) ----------


@pytest.mark.asyncio
async def test_ask_sends_permission_request_event(ctx, registry):
    from mcc_config import PermissionConfig

    checker = PermissionChecker(PermissionConfig(mode="default"))
    llm = MockLlm([
        _tool_turn("t1", "run_bash", {"command": "echo hi"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).permission(checker).build()

    events = []
    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)
    while not queue.empty():
        events.append(queue.get_nowait())

    perm_events = [e for e in events if isinstance(e, PermissionRequest)]
    assert len(perm_events) == 1
    assert "run_bash" in perm_events[0].message
    assert perm_events[0].tool_name == "run_bash"
    assert perm_events[0].tool_input == {"command": "echo hi"}

    tool_end = [e for e in events if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is True
    assert "permission denied by user" in tool_end.output


# ---------- Shell hook integration ----------


@pytest.mark.asyncio
async def test_shell_hook_integration_pre_tool(ctx, registry, tmp_path):
    log_file = tmp_path / "hook.log"
    script = tmp_path / "log_hook.sh"
    script.write_text(
        f"#!/usr/bin/env bash\n"
        f'echo "[$MCC_HOOK_TYPE] tool=$MCC_TOOL_NAME" >> "{log_file}"\n'
        f"exit 0\n"
    )
    os.chmod(script, 0o755)

    hooks = HookRegistry()
    hooks.add_shell(HookType.PRE_TOOL, script)

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "out.txt", "content": "x"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)

    assert log_file.exists()
    log_content = log_file.read_text()
    assert "pre_tool" in log_content
    assert "write_file" in log_content


# ---------- ShellHook matcher ----------


@pytest.mark.asyncio
async def test_shell_hook_matcher_wildcard_runs_for_any_tool(tmp_path: Path):
    script = tmp_path / "hook.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script, matcher="*")
    ctx = HookContext(hook_type=HookType.PRE_TOOL, tool_name="write_file", cwd=tmp_path)
    result = await hook.run(ctx)
    assert result.allow is True
    assert result.message == ""


@pytest.mark.asyncio
async def test_shell_hook_matcher_specific_tool(tmp_path: Path):
    script = tmp_path / "hook.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script, matcher="run_bash")

    matched = HookContext(hook_type=HookType.PRE_TOOL, tool_name="run_bash", cwd=tmp_path)
    result = await hook.run(matched)
    assert result.allow is False

    skipped = HookContext(hook_type=HookType.PRE_TOOL, tool_name="write_file", cwd=tmp_path)
    result = await hook.run(skipped)
    assert result.allow is True
    assert result.message == ""


@pytest.mark.asyncio
async def test_shell_hook_matcher_glob_pattern(tmp_path: Path):
    script = tmp_path / "hook.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n")
    os.chmod(script, 0o755)

    hook = ShellHook(HookType.PRE_TOOL, script, matcher="*_file")

    assert hook.matches("write_file") is True
    assert hook.matches("read_file") is True
    assert hook.matches("run_bash") is False


@pytest.mark.asyncio
async def test_shell_hook_matcher_empty_matches_all():
    hook = ShellHook(HookType.PRE_TOOL, "/dev/null", matcher="")
    assert hook.matches("anything") is True


@pytest.mark.asyncio
async def test_shell_hook_matcher_skips_in_agent_loop(ctx, registry, tmp_path):
    log_file = tmp_path / "hook.log"
    script = tmp_path / "log_hook.sh"
    script.write_text(f'#!/usr/bin/env bash\necho "$MCC_TOOL_NAME" >> "{log_file}"\nexit 0\n')
    os.chmod(script, 0o755)

    hooks = HookRegistry()
    hooks.add_shell(HookType.PRE_TOOL, script, matcher="run_bash")

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "out.txt", "content": "x"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hooks).build()

    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)

    assert not log_file.exists()


# ---------- Config-driven hook registration ----------


class TestBuildHooks:
    def test_empty_config_returns_none(self, tmp_path: Path):
        from mcc_cli import _build_hooks
        from mcc_config import Config

        assert _build_hooks(Config(), tmp_path) is None

    def test_registers_pre_and_post_hooks(self, tmp_path: Path):
        from mcc_cli import _build_hooks
        from mcc_config import Config, HookEntry, HooksConfig
        from mcc_harness import HookType

        config = Config(
            hooks=HooksConfig(
                pre_tool_use=[HookEntry(matcher="run_bash", command="hooks/audit.sh")],
                post_tool_use=[HookEntry(matcher="*", command="/abs/post.sh")],
            )
        )
        hook_registry = _build_hooks(config, tmp_path)
        assert hook_registry is not None
        assert len(hook_registry.get(HookType.PRE_TOOL)) == 1
        assert len(hook_registry.get(HookType.POST_TOOL)) == 1

    def test_relative_script_resolved_against_work_dir(self, tmp_path: Path):
        from mcc_cli import _build_hooks
        from mcc_config import Config, HookEntry, HooksConfig
        from mcc_harness import HookType

        config = Config(
            hooks=HooksConfig(
                pre_tool_use=[HookEntry(matcher="*", command="hooks/audit.sh")],
            )
        )
        hook_registry = _build_hooks(config, tmp_path)
        hook = hook_registry.get(HookType.PRE_TOOL)[0]
        assert hook._script == str(tmp_path / "hooks" / "audit.sh")

    def test_absolute_script_kept_as_is(self, tmp_path: Path):
        from mcc_cli import _build_hooks
        from mcc_config import Config, HookEntry, HooksConfig
        from mcc_harness import HookType

        config = Config(
            hooks=HooksConfig(
                pre_tool_use=[HookEntry(matcher="*", command="/usr/local/bin/audit.sh")],
            )
        )
        hook_registry = _build_hooks(config, tmp_path)
        hook = hook_registry.get(HookType.PRE_TOOL)[0]
        assert hook._script == "/usr/local/bin/audit.sh"

    def test_matcher_passed_through(self, tmp_path: Path):
        from mcc_cli import _build_hooks
        from mcc_config import Config, HookEntry, HooksConfig
        from mcc_harness import HookType

        config = Config(
            hooks=HooksConfig(
                pre_tool_use=[HookEntry(matcher="run_bash", command="/x.sh")],
            )
        )
        hook_registry = _build_hooks(config, tmp_path)
        hook = hook_registry.get(HookType.PRE_TOOL)[0]
        assert hook.matches("run_bash") is True
        assert hook.matches("write_file") is False


@pytest.mark.asyncio
async def test_config_hooks_end_to_end(ctx, registry, tmp_path, monkeypatch):
    log_file = tmp_path / "audit.log"
    script = tmp_path / "audit.sh"
    script.write_text(f'#!/usr/bin/env bash\necho "$MCC_TOOL_NAME" >> "{log_file}"\nexit 0\n')
    os.chmod(script, 0o755)

    from mcc_cli import _build_hooks
    from mcc_config import Config, HookEntry, HooksConfig

    config = Config(
        hooks=HooksConfig(
            pre_tool_use=[HookEntry(matcher="write_*", command=str(script))],
        )
    )
    hook_registry = _build_hooks(config, tmp_path)
    assert hook_registry is not None

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "out.txt", "content": "x"}),
        _tool_turn("t2", "read_file", {"path": "out.txt"}),
        single_text_turn("done"),
    ])
    agent = AgentLoopBuilder(llm, registry, ctx).hooks(hook_registry).build()

    queue: asyncio.Queue = asyncio.Queue()
    await agent.run_streaming("test", queue)

    logged = log_file.read_text().splitlines()
    assert logged == ["write_file"]
