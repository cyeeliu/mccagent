"""mcc 命令行入口。

解析参数并分发到 TUI 或 headless 模式。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

import click

from mcc_cli.events import AgentEventSource, EchoEventSource, EventSource
from mcc_cli.headless import run_headless


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("mcc-cli")
    except Exception:
        return "0.1.0"


def _system_prompt() -> str:
    return (
        "You are mini-claude-code, an enterprise-grade AI coding assistant. "
        "Examine files with read_file or list_dir before making changes. "
        "Use edit_file for targeted modifications. "
        "Use run_bash for shell tasks. "
        "Use grep to search code. "
        "Be concise and precise."
    )


def _build_hooks(config, work_dir: Path):
    """从配置构建钩子注册表：pre/post_tool_use 条目 → ShellHook。

    command 为脚本路径，相对路径基于工作目录解析。
    无钩子配置时返回 None。
    """
    from mcc_harness import HookRegistry, HookType

    entries = [
        (HookType.PRE_TOOL, e) for e in config.hooks.pre_tool_use
    ] + [
        (HookType.POST_TOOL, e) for e in config.hooks.post_tool_use
    ]
    if not entries:
        return None

    hook_registry = HookRegistry()
    for hook_type, entry in entries:
        script = Path(entry.command)
        if not script.is_absolute():
            script = work_dir / script
        hook_registry.add_shell(hook_type, script, matcher=entry.matcher)
    return hook_registry


def _resolve_model(
    config, kind: str = "main", explicit: str | None = None
) -> str:
    """按用途解析模型：显式参数 > 环境变量 MODEL > 配置该类型 > 配置 main。

    main / subagent / summarize 等类型共用此逻辑；
    配置只写了 main 时，其余类型一律回落 main。
    """
    if explicit:
        return explicit
    env = os.environ.get("MODEL")
    if env:
        return env
    configured = getattr(config.model, kind)
    if configured:
        return configured
    return config.model.main


def _build_source(
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    history: list | None = None,
    session_id: str | None = None,
) -> EventSource:
    """构建事件源：mock 模式用 MockLlm，否则用真实 LLM。

    同时接线会话录制（JSONL 落盘）与长期记忆
    （save_memory / read_memory 工具 + system prompt 索引注入）。
    """
    from mcc_config import ensure_default_config
    from mcc_core import ToolContext
    from mcc_harness import (
        AgentLoopBuilder,
        PermissionChecker,
        SpawnSubagentTool,
        SubagentRunner,
        default_presets,
    )
    from mcc_session import MarkdownMemoryStore, SessionRecorder, memory_tools
    from mcc_tools import default_registry

    config = ensure_default_config(work_dir)
    sid = session_id or uuid.uuid4().hex[:8]
    ctx = ToolContext(cwd=work_dir, session_id=sid)
    registry = default_registry()

    system = _system_prompt()
    store = MarkdownMemoryStore()
    for tool in memory_tools(store):
        registry.register(tool)
    index = store.index_text()
    if index:
        system = f"{system}\n\n# Memory Index\n{index}"

    if mock or mock_script:
        from mcc_llm.mock import MockLlm

        if mock_script:
            turns = _parse_mock_script(mock_script)
        else:
            from mcc_llm.mock import single_text_turn

            turns = [single_text_turn("[mock] No --mock-script provided; echoing.")]
        llm = MockLlm(turns)
    else:
        try:
            from mcc_llm import auto_provider_from_env

            llm = auto_provider_from_env()
        except RuntimeError as e:
            click.echo(f"Error: {e}\nUse --mock for testing without API keys.", err=True)
            sys.exit(2)

    hook_registry = _build_hooks(config, work_dir)

    subagent_model = _resolve_model(config, "subagent")
    summarize_model = _resolve_model(config, "summarize")
    runner = SubagentRunner(
        llm=llm,
        registry=registry,
        concurrency=4,
        hooks=hook_registry,
    )
    registry.register(
        SpawnSubagentTool(runner, default_presets(subagent_model, summarize_model))
    )

    recorder = SessionRecorder.open_sync(sid, work_dir)
    builder = AgentLoopBuilder(llm, registry, ctx).system(system).recorder(recorder)

    builder = builder.model(_resolve_model(config, "main", model))
    builder = builder.summarize_model(summarize_model)

    checker = PermissionChecker(config.permissions)
    builder = builder.permission(checker)

    builder = builder.max_cost_usd(config.budget.max_usd_per_session)
    builder = builder.max_iterations(config.budget.max_iterations)

    if history:
        builder = builder.history(history)

    if hook_registry is not None:
        builder = builder.hooks(hook_registry)

    agent = builder.build()
    return AgentEventSource(agent)


def _parse_mock_script(path: str) -> list:
    """从 JSON 文件加载 MockLlm 脚本。"""
    from mcc_core import Usage
    from mcc_llm import MessageStop, TextDelta, ToolUseInputDelta, ToolUseStart

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = []
    for turn in data:
        events = []
        for ev in turn:
            kind = ev["kind"]
            if kind == "text":
                events.append(TextDelta(text=ev.get("text", "")))
            elif kind == "tool_use_start":
                events.append(ToolUseStart(id=ev["id"], name=ev["name"]))
            elif kind == "tool_use_input":
                events.append(ToolUseInputDelta(partial_json=json.dumps(ev["input"])))
            elif kind == "message_stop":
                events.append(
                    MessageStop(
                        stop_reason=ev.get("stop_reason", "end_turn"),
                        usage=Usage(**ev.get("usage", {})),
                    )
                )
        turns.append(events)
    return turns


def _history_events(history: list, cost_usd: float = 0.0) -> list:
    """把恢复的消息历史转换为 TUI 重放事件序列。

    历史只传给 AgentLoop 时 LLM 具备上下文，但渲染层从未
    收到事件，界面会是空白；此处按 user/assistant/tool 消息
    逐条还原轨迹，末尾以 TurnEnd 收尾（携带会话累计成本）。
    """
    from mcc_core import Role, TextBlock, ToolResultBlock, ToolUseBlock
    from mcc_core.event import TextDelta, ToolCallEnd, ToolCallStart, TurnEnd, UserEcho

    events = []
    for m in history:
        if m.role == Role.user:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(UserEcho(text=b.text))
        elif m.role == Role.assistant:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(TextDelta(text=b.text))
                elif isinstance(b, ToolUseBlock):
                    events.append(ToolCallStart(id=b.id, name=b.name, args_preview=""))
        elif m.role == Role.tool:
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    events.append(
                        ToolCallEnd(
                            id=b.tool_use_id,
                            output=b.content[:500],
                            is_error=b.is_error,
                        )
                    )
    events.append(TurnEnd(cost_usd=cost_usd))
    return events


async def _run_tui_mode(
    source: EventSource, replay: list | None = None, session_id: str | None = None
) -> int:
    from mcc_tui import TuiHandles, run_tui

    event_queue: asyncio.Queue = asyncio.Queue()
    input_queue: asyncio.Queue = asyncio.Queue()
    permission_queue: asyncio.Queue = asyncio.Queue()
    session_short_id = session_id or uuid.uuid4().hex[:8]

    agent = getattr(source, "agent", None)
    if agent is not None:
        agent.permission_responses = permission_queue

    for ev in replay or []:
        event_queue.put_nowait(ev)

    async def agent_worker() -> None:
        while True:
            user_input = await input_queue.get()
            try:
                async for ev in source.run(user_input):
                    await event_queue.put(ev)
            except Exception as exc:
                from mcc_core import Error as ErrorEvent

                await event_queue.put(ErrorEvent(text=str(exc)))

    worker = asyncio.create_task(agent_worker())
    try:
        handles = TuiHandles(
            events=event_queue,
            input_tx=input_queue,
            session_short_id=session_short_id,
            permission_tx=permission_queue,
        )
        await run_tui(handles)
    finally:
        worker.cancel()
    return 0


def _parse_days(value: str | None) -> int | None:
    """解析 Nd 形式的天数，非法返回 None。"""
    if value is None:
        return None
    m = re.fullmatch(r"(\d+)d", value)
    return int(m.group(1)) if m else None


def _run_sessions_cmd(args: tuple[str, ...], older_than: str | None) -> None:
    """sessions 子命令：list / delete / purge。"""
    from mcc_session import (
        SessionNotFoundError,
        delete_session,
        list_sessions,
        purge_sessions,
    )

    sub = args[0] if args else "list"

    if sub == "list":
        metas = list_sessions()
        if not metas:
            click.echo("No sessions found.")
            return
        click.echo(f"{'ID':<10} {'STARTED':<17} {'TURNS':>5} {'COST':>9}  TITLE")
        for m in metas:
            started = m.started_at.strftime("%Y-%m-%d %H:%M")
            cost = f"${m.cost_usd:.4f}"
            title = m.title or ""
            click.echo(f"{m.id:<10} {started:<17} {m.turns:>5} {cost:>9}  {title}".rstrip())
        return

    if sub == "delete":
        if len(args) < 2:
            click.echo("Usage: mcc sessions delete <session-id>", err=True)
            sys.exit(2)
        try:
            meta = delete_session(args[1])
        except SessionNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo(f"deleted session {meta.id} ({meta.turns} turn(s))")
        return

    if sub == "purge":
        days = _parse_days(older_than)
        if days is None:
            click.echo("Usage: mcc sessions purge --older-than <Nd>", err=True)
            sys.exit(2)
        removed = purge_sessions(days)
        click.echo(f"purged {removed} session(s)")
        return

    click.echo(f"Unknown sessions subcommand: {sub}", err=True)
    sys.exit(2)


def _run_resume_cmd(
    args: tuple[str, ...],
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
) -> None:
    """resume 子命令：加载历史会话并继续对话。"""
    from mcc_session import SessionNotFoundError, load_session

    if not args:
        click.echo("Usage: mcc resume <session-id>", err=True)
        sys.exit(2)

    try:
        meta, history = load_session(args[0])
    except SessionNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not (mock or mock_script or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        click.echo(
            "Error: no LLM API key found. Use --mock for testing without API keys.",
            err=True,
        )
        sys.exit(2)

    click.echo(f"Resuming session {meta.id} ({meta.turns} turn(s), "
               f"${meta.cost_usd:.4f})" + (f" — {meta.title}" if meta.title else ""),
               err=True)
    source = _build_source(
        work_dir, model, mock, mock_script, history=history, session_id=meta.id
    )
    replay = _history_events(history, cost_usd=meta.cost_usd)
    sys.exit(
        asyncio.run(_run_tui_mode(source, replay=replay, session_id=meta.id))
    )


@click.command()
@click.option("-p", "--prompt", "prompt", default=None, help="单次 headless 模式")
@click.option("--cwd", "cwd", default=None, envvar="MCC_PROJECT", help="工作目录")
@click.option("--model", "model", default=None, envvar="MODEL", help="覆盖模型")
@click.option("--quiet", "quiet", is_flag=True, help="只输出最终答案")
@click.option("--mock", "mock", is_flag=True, help="使用 Mock LLM（无需 API key）")
@click.option(
    "--mock-script",
    "mock_script",
    default=None,
    help="MockLlm 脚本 JSON 文件路径",
)
@click.option(
    "--older-than",
    "older_than",
    default=None,
    hidden=True,
    help="sessions purge：清理 N 天前的会话（如 30d）",
)
@click.argument(
    "cmd",
    required=False,
    type=click.Choice(["tui", "config", "version", "resume", "sessions"]),
)
@click.argument("args", nargs=-1)
def main(
    cmd: str | None,
    prompt: str | None,
    cwd: str | None,
    model: str | None,
    quiet: bool,
    mock: bool,
    mock_script: str | None,
    older_than: str | None,
    args: tuple[str, ...],
) -> None:
    """mini-claude-code — 企业级 AI 编码助手。"""
    work_dir = Path(cwd) if cwd else Path.cwd()

    if cmd == "version":
        click.echo(f"mcc {_version()}")
        return

    if cmd == "config":
        from mcc_config import load

        config = asyncio.run(load(work_dir))
        click.echo(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

    if cmd == "sessions":
        _run_sessions_cmd(args, older_than)
        return

    if cmd == "resume":
        _run_resume_cmd(args, work_dir, model, mock, mock_script)
        return

    if mock or mock_script:
        source = _build_source(work_dir, model, mock, mock_script)
    elif os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        source = _build_source(work_dir, model, False, None)
    else:
        source = EchoEventSource()

    try:
        if cmd == "tui" or (cmd is None and prompt is None):
            sys.exit(asyncio.run(_run_tui_mode(source)))
        elif prompt is not None:
            sys.exit(asyncio.run(run_headless(source, prompt, quiet)))
    except KeyboardInterrupt:
        sys.exit(0)


__all__ = ["main"]
__version__ = "0.1.0"
