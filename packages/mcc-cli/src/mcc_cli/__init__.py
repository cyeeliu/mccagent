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

from mcc_cli.events import EchoEventSource, EventSource
from mcc_cli.headless import run_headless

# 装配层已拆至 mcc_cli.wiring（各 _wire_* 子系统构造 + build_source
# 编排）；此处 re-export 保持 from mcc_cli import ... 兼容。
from mcc_cli.wiring import _build_hooks as _build_hooks
from mcc_cli.wiring import _parse_mock_script as _parse_mock_script
from mcc_cli.wiring import _register_builtin_workflows as _register_builtin_workflows
from mcc_cli.wiring import _system_prompt as _system_prompt
from mcc_cli.wiring import build_source


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


def _build_source(
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    history: list | None = None,
    session_id: str | None = None,
    no_compact: bool = False,
    no_skills: bool = False,
    llm_base_url: str | None = None,
) -> EventSource:
    """构建事件源（薄壳）：实际装配在 mcc_cli.wiring.build_source。"""
    return build_source(
        work_dir,
        model,
        mock,
        mock_script,
        history=history,
        session_id=session_id,
        no_compact=no_compact,
        no_skills=no_skills,
        llm_base_url=llm_base_url,
    )

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
                    events.append(ToolCallStart(
                        id=b.id, name=b.name,
                        args_preview=json.dumps(b.input, ensure_ascii=False)
                        if b.input else "",
                    ))
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


def _dump_metrics_snapshot(sink, work_dir: Path | None) -> None:
    """TUI 退出时落盘指标快照并向 stderr 打一行汇总。"""
    import sys as _sys

    snapshot = sink.metrics.snapshot()
    target_dir = work_dir or Path.cwd()
    try:
        out = target_dir / ".mcc" / "last_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    except OSError:
        return
    llm_calls = sum(
        c["value"] for c in snapshot["counters"]
        if c["name"] == "agent_llm_calls_total"
    )
    latency = next(
        (h for h in snapshot["histograms"]
         if h["name"] == "agent_llm_latency_ms"),
        None,
    )
    cache = next(
        (h for h in snapshot["histograms"]
         if h["name"] == "agent_cache_hit_rate"),
        None,
    )
    parts = [f"llm_calls={int(llm_calls)}"]
    if latency:
        parts.append(f"p95={latency['p95']:.0f}ms")
    if cache:
        parts.append(f"cache_hit_rate={cache['sum'] / max(cache['count'], 1):.2f}")
    print(f"[observability] {' '.join(parts)}", file=_sys.stderr)


async def _run_tui_mode(
    source: EventSource,
    replay: list | None = None,
    session_id: str | None = None,
    work_dir: Path | None = None,
) -> int:
    from mcc_harness import context_window
    from mcc_tui import TuiHandles, run_tui
    from mcc_tui.completions import make_provider

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
            team_mgr = getattr(source, "team_manager", None)
            if team_mgr is not None:
                inbox = await team_mgr.consume_lead_inbox()
                if inbox:
                    from mcc_harness import format_team_events

                    input_queue.put_nowait(format_team_events(inbox))

    worker = asyncio.create_task(agent_worker())
    cron = getattr(source, "cron_scheduler", None)
    if cron is not None:
        await cron.start()
    wf_tool = getattr(source, "workflow_tool", None)
    if wf_tool is not None:
        wf_tool.set_event_sink(event_queue)
    try:
        slash_registry = getattr(source, "slash_registry", None)
        commands = slash_registry.list_commands() if slash_registry is not None else []
        agent_obj = getattr(source, "agent", None)
        model_id = getattr(agent_obj, "model", "") or ""
        handles = TuiHandles(
            events=event_queue,
            input_tx=input_queue,
            session_short_id=session_short_id,
            permission_tx=permission_queue,
            completions=make_provider(commands),
            model_id=model_id,
            context_limit=context_window(model_id) if model_id else 0,
        )
        await run_tui(handles)
    finally:
        worker.cancel()
        if cron is not None:
            await cron.stop()
        sink = getattr(source, "observability_sink", None)
        if sink is not None:
            _dump_metrics_snapshot(sink, work_dir)
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


def _run_onboarding_wizard() -> bool:
    """交互式首启配置向导：provider/网关/key/模型写入 home settings。

    返回 True 表示配置完成可继续正常启动；用户中断返回 False。
    其余配置一律保持默认，后续可用 /config 修改。
    """
    from mcc_config import update_home_setting

    click.echo("欢迎使用 mini-claude-code！首次运行需要配置 LLM 连接。\n")
    try:
        kind = click.prompt(
            "Provider 类型（anthropic 官方 / gateway 自托管网关）",
            default="anthropic",
            type=click.Choice(["anthropic", "gateway"], case_sensitive=False),
        )
        if kind.lower() == "gateway":
            base_url = click.prompt(
                "Base URL", default="http://localhost:8000/v1"
            )
            update_home_setting(["llm", "base_url"], base_url)
        api_key = click.prompt(
            "API key（自托管网关可留空）",
            default="",
            show_default=False,
            hide_input=True,
        )
        if api_key.strip():
            update_home_setting(["llm", "api_key"], api_key.strip())
        model_id = click.prompt("主模型 ID", default="claude-opus-4-8")
        update_home_setting(["model", "main"], model_id.strip())
    except (click.Abort, EOFError):
        click.echo("\n已取消向导。", err=True)
        return False
    click.echo("配置已写入 ~/.mcc/settings.json，其余保持默认。\n")
    return True


def _print_onboarding() -> None:
    """无 API key 时的首次运行引导。"""
    click.echo(
        "欢迎使用 mini-claude-code！\n"
        "\n"
        "尚未配置 LLM API key，请先设置其一：\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  export OPENAI_API_KEY=sk-...          # OpenAI 兼容端点\n"
        "\n"
        "自托管网关（vLLM / SGLang / Ollama 等）无需外部 key：\n"
        "  mcc --llm-base-url http://gateway:8000/v1 -p \"hi\"\n"
        "\n"
        "也可写入项目根目录 .env 文件（参考 .env.example）。\n"
        "无 key 体验完整链路：mcc --mock\n"
        "查阅文档：https://github.com/cyeeliu/mccagent",
        err=True,
    )


_COMPLETION_SHELLS = ("bash", "zsh", "fish")


def _run_completion_cmd(shell: str) -> None:
    """completion 子命令：输出 shell 补全脚本。"""
    from click.shell_completion import get_completion_class

    if shell not in _COMPLETION_SHELLS:
        click.echo(
            f"Unsupported shell: {shell} ({'/'.join(_COMPLETION_SHELLS)})", err=True
        )
        sys.exit(2)
    comp_cls = get_completion_class(shell)
    assert comp_cls is not None
    comp = comp_cls(main, {}, "mcc", "_MCC_COMPLETE")
    script = getattr(comp, "get_completion_script", None)
    click.echo(script() if script else comp.source())


def _latest_pypi_version(package: str = "mcc-cli") -> str | None:
    """查询 PyPI 上的最新发布版本，网络失败返回 None。"""
    import httpx

    try:
        resp = httpx.get(
            f"https://pypi.org/pypi/{package}/json", timeout=5.0
        )
        resp.raise_for_status()
        return resp.json()["info"]["version"]
    except Exception:
        return None


def _parse_version(v: str) -> tuple[int, ...]:
    """把 'a.b.c' 解析为可比较的整数元组，每段只取前导数字。"""
    parts = []
    for seg in v.strip().lstrip("v").split("."):
        m = re.match(r"\d+", seg)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def _is_newer_version(latest: str, current: str) -> bool:
    return _parse_version(latest) > _parse_version(current)


def _run_update_cmd() -> None:
    """update 子命令：比对 PyPI 最新版本并提示升级方式。"""
    current = _version()
    click.echo(f"mcc {current}")
    latest = _latest_pypi_version()
    if latest is None:
        click.echo("Could not check for updates (network unreachable).", err=True)
        return
    if _is_newer_version(latest, current):
        click.echo(f"New version available: {latest}")
        click.echo("Upgrade with:\n  uv tool upgrade mcc-cli\n  # or: pipx upgrade mcc-cli")
    else:
        click.echo("You are on the latest version.")


def _run_resume_cmd(
    args: tuple[str, ...],
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    llm_base_url: str | None = None,
    no_compact: bool = False,
    no_skills: bool = False,
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

    if not (mock or mock_script or llm_base_url
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        from mcc_config import read_home_settings

        home_llm = read_home_settings().get("llm") or {}
        if not (home_llm.get("api_key") or home_llm.get("base_url")):
            _print_onboarding()
            sys.exit(2)

    click.echo(f"Resuming session {meta.id} ({meta.turns} turn(s), "
               f"${meta.cost_usd:.4f})" + (f" — {meta.title}" if meta.title else ""),
               err=True)
    source = _build_source(
        work_dir,
        model,
        mock,
        mock_script,
        history=history,
        session_id=meta.id,
        llm_base_url=llm_base_url,
        no_compact=no_compact,
        no_skills=no_skills,
    )
    replay = _history_events(history, cost_usd=meta.cost_usd)
    sys.exit(
        asyncio.run(_run_tui_mode(source, replay=replay, session_id=meta.id, work_dir=work_dir))
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
    "--no-compact",
    "no_compact",
    is_flag=True,
    help="禁用上下文自动压缩（长会话可能因上下文超限失败）",
)
@click.option(
    "--no-skills",
    "no_skills",
    is_flag=True,
    help="禁用技能与 slash 命令加载",
)
@click.option(
    "--llm-base-url",
    "llm_base_url",
    default=None,
    envvar="MCC_LLM_BASE_URL",
    help="OpenAI 兼容自托管网关地址（vLLM / SGLang / Ollama 等）",
)
@click.option(
    "--no-wizard",
    "no_wizard",
    is_flag=True,
    help="跳过首次运行的交互式配置向导",
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
    type=click.Choice(
        ["tui", "config", "version", "resume", "sessions", "completion", "update"]
    ),
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
    llm_base_url: str | None,
    no_compact: bool,
    no_skills: bool,
    no_wizard: bool,
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

    if cmd == "completion":
        if not args:
            click.echo("Usage: mcc completion <bash|zsh|fish>", err=True)
            sys.exit(2)
        _run_completion_cmd(args[0])
        return

    if cmd == "update":
        _run_update_cmd()
        return

    if cmd == "resume":
        _run_resume_cmd(args, work_dir, model, mock, mock_script, llm_base_url,
                         no_compact, no_skills)
        return

    has_llm = bool(
        mock or mock_script or llm_base_url
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not has_llm:
        from mcc_config import read_home_settings

        home_llm = read_home_settings().get("llm") or {}
        if home_llm.get("api_key") or home_llm.get("base_url"):
            has_llm = True

    if has_llm:
        source = _build_source(work_dir, model, mock, mock_script,
                               llm_base_url=llm_base_url,
                               no_compact=no_compact, no_skills=no_skills)
    elif prompt is not None:
        _print_onboarding()
        sys.exit(2)
    elif not no_wizard and _run_onboarding_wizard():
        # TUI 首启：向导写入 home settings 后直接正常构建启动
        source = _build_source(work_dir, model, mock, mock_script,
                               llm_base_url=llm_base_url,
                               no_compact=no_compact, no_skills=no_skills)
    else:
        _print_onboarding()
        source = EchoEventSource()

    try:
        if cmd == "tui" or (cmd is None and prompt is None):
            sys.exit(asyncio.run(_run_tui_mode(source, work_dir=work_dir)))
        elif prompt is not None:
            sys.exit(asyncio.run(run_headless(source, prompt, quiet)))
    except KeyboardInterrupt:
        sys.exit(0)


__all__ = ["main"]
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("mcc-cli")
except Exception:
    __version__ = "0.1.0"
