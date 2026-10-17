"""权限交互测试：AgentLoop 等待用户决策 + TUI 按键回路。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_config import PermissionConfig
from mcc_core import (
    PermissionDecision,
    PermissionRequest,
    ToolContext,
    Usage,
)
from mcc_core.event import Notice, ToolCallEnd
from mcc_harness import (
    AgentLoopBuilder,
    Allow,
    BashAction,
    PathAction,
    PermissionChecker,
)
from mcc_harness import (
    PermissionRequest as PermissionReq,
)
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import default_registry
from mcc_tui import TuiHandles, _handle_key
from mcc_tui.app import App
from mcc_tui.keys import Key


def _tool_turn(tool_id: str, tool_name: str, tool_input: dict):
    return [
        ToolUseStart(id=tool_id, name=tool_name),
        ToolUseInputDelta(partial_json=json.dumps(tool_input)),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


def _drain(queue: asyncio.Queue) -> list:
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


_STOP = object()


def _start_responder(evq: asyncio.Queue, responses: asyncio.Queue, decide):
    """启动 dispatcher：把事件复制到 sink 列表，同时响应权限请求。"""
    sink: list = []

    async def responder():
        while True:
            ev = await evq.get()
            if ev is _STOP:
                return
            sink.append(ev)
            if isinstance(ev, PermissionRequest):
                responses.put_nowait(decide(ev))

    return sink, asyncio.create_task(responder())


async def _stop_responder(evq: asyncio.Queue, task: asyncio.Task):
    evq.put_nowait(_STOP)
    await task


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="perm-test")


@pytest.fixture
def registry():
    return default_registry()


@pytest.fixture
def checker():
    return PermissionChecker(PermissionConfig(mode="default"))


def _agent(llm, ctx, registry, checker, responses=None, **kwargs):
    builder = (
        AgentLoopBuilder(llm, registry, ctx)
        .permission(checker)
        .permission_responses(responses)
    )
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


# ---------- PermissionChecker.allow_always ----------


class TestAllowAlways:
    def test_bash_exact_command_allowed(self, checker):
        checker.allow_always("Bash", BashAction(cmd="git status"))
        assert isinstance(
            checker.check(PermissionReq(category="Bash", action=BashAction(cmd="git status"))),
            Allow,
        )

    def test_bash_different_command_still_asks(self, checker):
        checker.allow_always("Bash", BashAction(cmd="git status"))
        decision = checker.check(
            PermissionReq(category="Bash", action=BashAction(cmd="git push"))
        )
        assert not isinstance(decision, Allow)

    def test_path_allowed(self, checker):
        checker.allow_always("Write", PathAction(path="src/main.py"))
        assert isinstance(
            checker.check(PermissionReq(category="Write", action=PathAction(path="src/main.py"))),
            Allow,
        )

    def test_empty_command_not_added(self, checker):
        before = len(checker._allow_bash)
        checker.allow_always("Bash", BashAction(cmd="  "))
        assert len(checker._allow_bash) == before


# ---------- AgentLoop._await_permission_decision ----------


@pytest.mark.asyncio
async def test_allow_decision_executes_tool(ctx, registry, checker, tmp_path):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "allowed.txt", "content": "ok"}),
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, responses)

    sink, resp_task = _start_responder(
        evq,
        responses,
        lambda ev: PermissionDecision(request_id=ev.id, action="allow"),
    )
    await agent.run_streaming("go", evq)
    await _stop_responder(evq, resp_task)

    assert (tmp_path / "allowed.txt").read_text() == "ok"
    tool_end = [e for e in sink if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is False


@pytest.mark.asyncio
async def test_deny_decision_blocks_tool(ctx, registry, checker, tmp_path):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "blocked.txt", "content": "x"}),
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, responses)

    sink, resp_task = _start_responder(
        evq,
        responses,
        lambda ev: PermissionDecision(request_id=ev.id, action="deny"),
    )
    await agent.run_streaming("go", evq)
    await _stop_responder(evq, resp_task)

    assert not (tmp_path / "blocked.txt").exists()
    tool_end = [e for e in sink if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is True
    assert "permission denied by user" in tool_end.output


@pytest.mark.asyncio
async def test_feedback_decision_returns_comment(ctx, registry, checker):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "run_bash", {"command": "rm -rf /"}),
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, responses)

    def decide(ev):
        return PermissionDecision(
            request_id=ev.id,
            action="feedback",
            feedback="use trash instead of rm",
        )

    sink, resp_task = _start_responder(evq, responses, decide)
    await agent.run_streaming("go", evq)
    await _stop_responder(evq, resp_task)

    tool_end = [e for e in sink if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is True
    assert "use trash instead of rm" in tool_end.output


@pytest.mark.asyncio
async def test_allow_always_adds_rule_and_second_call_skips_ask(
    ctx, registry, checker
):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()

    turn_write = _tool_turn("t1", "write_file", {"path": "a.txt", "content": "A"})
    turn_bash = _tool_turn("t2", "run_bash", {"command": "echo twice"})
    llm = MockLlm([
        turn_bash,
        turn_write,
        # 相同命令第二次调用：allow_always 已加规则，应跳过 Ask 直接执行
        turn_bash,
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, responses)

    sink, resp_task = _start_responder(
        evq,
        responses,
        lambda ev: PermissionDecision(request_id=ev.id, action="allow_always"),
    )
    await agent.run_streaming("go", evq)
    await _stop_responder(evq, resp_task)

    perm_events = [e for e in sink if isinstance(e, PermissionRequest)]
    assert len(perm_events) == 2
    tool_ends = [e for e in sink if isinstance(e, ToolCallEnd)]
    assert all(not te.is_error for te in tool_ends)


@pytest.mark.asyncio
async def test_stale_decision_ignored(ctx, registry, checker):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "c.txt", "content": "C"}),
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, responses)

    state = {"seen": False}

    def decide(ev):
        if state["seen"]:
            return PermissionDecision(request_id=ev.id, action="allow")
        state["seen"] = True
        # 先放一个过期决策（id 不匹配），再放正确决策
        responses.put_nowait(
            PermissionDecision(request_id=ev.id + 999, action="deny")
        )
        return PermissionDecision(request_id=ev.id, action="allow")

    sink, resp_task = _start_responder(evq, responses, decide)
    await agent.run_streaming("go", evq)
    await _stop_responder(evq, resp_task)

    tool_end = [e for e in sink if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is False


@pytest.mark.asyncio
async def test_timeout_defaults_to_deny(ctx, registry, checker, tmp_path):
    responses: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "never.txt", "content": "x"}),
        single_text_turn("done"),
    ])
    agent = _agent(
        llm, ctx, registry, checker, responses, permission_timeout_secs=0.05
    )

    await agent.run_streaming("go", evq)

    assert not (tmp_path / "never.txt").exists()
    events = _drain(evq)
    tool_end = [e for e in events if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is True


@pytest.mark.asyncio
async def test_headless_without_channel_denies(ctx, registry, checker, tmp_path):
    evq: asyncio.Queue = asyncio.Queue()
    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "no.txt", "content": "x"}),
        single_text_turn("done"),
    ])
    agent = _agent(llm, ctx, registry, checker, None)

    await agent.run_streaming("go", evq)

    assert not (tmp_path / "no.txt").exists()


# ---------- TUI 按键回路 ----------


def _handles(tmp_path: Path):
    events: asyncio.Queue = asyncio.Queue()
    input_tx: asyncio.Queue = asyncio.Queue()
    permission_tx: asyncio.Queue = asyncio.Queue()
    return TuiHandles(
        events=events,
        input_tx=input_tx,
        session_short_id="t",
        permission_tx=permission_tx,
    ), permission_tx


class TestTuiPermissionKeys:
    def test_y_sends_allow(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "y")
        decision = permission_tx.get_nowait()
        assert decision.request_id == 7
        assert decision.action == "allow"
        assert app.pending_permission is None

    def test_a_sends_allow_always(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "a")
        decision = permission_tx.get_nowait()
        assert decision.action == "allow_always"

    def test_n_sends_deny(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "n")
        decision = permission_tx.get_nowait()
        assert decision.action == "deny"

    def test_escape_sends_deny(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, Key.escape)
        decision = permission_tx.get_nowait()
        assert decision.action == "deny"

    def test_e_enters_feedback_mode(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "e")
        assert app.permission_feedback_mode is True
        assert permission_tx.empty()

    def test_feedback_typing_and_submit(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "e")
        for ch in "nope please":
            _handle_key(app, handles, ch)
        assert app.permission_feedback == "nope please"
        _handle_key(app, handles, Key.backspace)
        assert app.permission_feedback == "nope pleas"
        _handle_key(app, handles, Key.enter)
        decision = permission_tx.get_nowait()
        assert decision.action == "feedback"
        assert decision.feedback == "nope pleas"
        assert app.permission_feedback_mode is False

    def test_feedback_escape_returns_to_choice(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "e")
        _handle_key(app, handles, "abc")
        _handle_key(app, handles, Key.escape)
        assert app.permission_feedback_mode is False
        assert app.permission_feedback == ""
        assert app.pending_permission is not None
        assert permission_tx.empty()

    def test_decision_notice_pushed(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=7, message="run_bash: echo hi"))
        handles, _ = _handles(Path("."))
        _handle_key(app, handles, "y")
        notices = [it for it in app.items if it.kind == "notice"]
        assert any("allowed" in it.text for it in notices)

    def test_queued_request_shown_after_resolve(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=1, message="first?"))
        app.apply_event(PermissionRequest(id=2, message="second?"))
        handles, permission_tx = _handles(Path("."))
        _handle_key(app, handles, "n")
        assert app.pending_permission is not None
        assert app.pending_permission.id == 2
        assert permission_tx.get_nowait().request_id == 1

    def test_no_permission_tx_still_resolves(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=1, message="first?"))
        handles = TuiHandles(
            events=asyncio.Queue(),
            input_tx=asyncio.Queue(),
            session_short_id="t",
        )
        _handle_key(app, handles, "y")
        assert app.pending_permission is None

    def test_other_keys_ignored_during_permission(self):
        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=1, message="first?"))
        handles, permission_tx = _handles(Path("."))
        for key in ("x", "z", Key.up, Key.down):
            _handle_key(app, handles, key)
        assert app.pending_permission is not None
        assert app.input == ""
        assert permission_tx.empty()


# ---------- 渲染 ----------


class TestPermissionRender:
    def test_pending_renders_permission_panel(self):
        from mcc_tui.render import build_renderable
        from rich.console import Console

        app = App(session_short_id="t")
        app.apply_event(
            PermissionRequest(id=1, message="run_bash: confirmation required")
        )
        console = Console(width=100, height=30)
        layout = build_renderable(app, console)
        # input 区（第二个子布局）渲染为权限确认面板
        input_area = layout.children[1]
        rendered = Console().render(input_area.renderable)
        text = "".join(seg.text for seg in rendered)
        assert "permission required" in text
        assert "[y]" in text and "[a]" in text and "[n]" in text and "[e]" in text

    def test_feedback_mode_renders_feedback_panel(self):
        from mcc_tui.render import build_renderable
        from rich.console import Console

        app = App(session_short_id="t")
        app.apply_event(PermissionRequest(id=1, message="run_bash: confirm?"))
        app.permission_feedback_mode = True
        app.permission_feedback = "my comment"
        console = Console(width=100, height=30)
        layout = build_renderable(app, console)
        input_area = layout.children[1]
        rendered = Console().render(input_area.renderable)
        text = "".join(seg.text for seg in rendered)
        assert "feedback" in text
        assert "my comment" in text

    def test_normal_input_when_no_permission(self):
        from mcc_tui.render import build_renderable
        from rich.console import Console

        app = App(session_short_id="t")
        app.input = "hello"
        console = Console(width=100, height=30)
        layout = build_renderable(app, console)
        input_area = layout.children[1]
        rendered = Console().render(input_area.renderable)
        text = "".join(seg.text for seg in rendered)
        assert "hello" in text


# ---------- e2e：AgentEventSource + permission queue ----------


@pytest.mark.asyncio
async def test_e2e_tui_permission_flow(tmp_path: Path):
    from mcc_cli.events import AgentEventSource

    ctx = ToolContext(cwd=tmp_path, session_id="e2e")
    checker = PermissionChecker(PermissionConfig(mode="default"))
    permission_queue: asyncio.Queue = asyncio.Queue()
    evq: asyncio.Queue = asyncio.Queue()

    llm = MockLlm([
        _tool_turn("t1", "write_file", {"path": "e2e.txt", "content": "E2E"}),
        single_text_turn("written"),
    ])
    agent = (
        AgentLoopBuilder(llm, default_registry(), ctx)
        .permission(checker)
        .permission_responses(permission_queue)
        .build()
    )
    source = AgentEventSource(agent)

    async def run_agent():
        async for ev in source.run("create file"):
            await evq.put(ev)
            if isinstance(ev, PermissionRequest):
                permission_queue.put_nowait(
                    PermissionDecision(request_id=ev.id, action="allow")
                )

    await asyncio.wait_for(run_agent(), timeout=5)
    assert (tmp_path / "e2e.txt").read_text() == "E2E"

    events = _drain(evq)
    perm = [e for e in events if isinstance(e, PermissionRequest)]
    assert len(perm) == 1
    assert perm[0].tool_name == "write_file"
    tool_end = [e for e in events if isinstance(e, ToolCallEnd)][0]
    assert tool_end.is_error is False
    notices = [e for e in events if isinstance(e, Notice)]
    assert not any("denied" in n.text for n in notices)
