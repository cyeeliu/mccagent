"""mcc 命令行入口。

解析参数并分发到 TUI 或 headless 模式。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import click

from mcc_cli.events import EchoEventSource, EventSource
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


async def _run_tui_mode(source: EventSource) -> int:
    from mcc_tui import TuiHandles, run_tui

    event_queue: asyncio.Queue = asyncio.Queue()
    input_queue: asyncio.Queue = asyncio.Queue()
    session_id = uuid.uuid4().hex[:8]

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
            events=event_queue, input_tx=input_queue, session_short_id=session_id
        )
        await run_tui(handles)
    finally:
        worker.cancel()
    return 0


@click.command()
@click.option("-p", "--prompt", "prompt", default=None, help="单次 headless 模式")
@click.option("--cwd", "cwd", default=None, envvar="MCC_PROJECT", help="工作目录")
@click.option("--model", "model", default=None, envvar="MODEL", help="覆盖模型")
@click.option("--quiet", "quiet", is_flag=True, help="只输出最终答案")
@click.argument(
    "cmd", required=False, type=click.Choice(["tui", "config", "version"])
)
def main(
    cmd: str | None,
    prompt: str | None,
    cwd: str | None,
    model: str | None,
    quiet: bool,
) -> None:
    """mini-claude-code — 企业级 AI 编码助手。"""
    _load_dotenv()
    work_dir = Path(cwd) if cwd else Path.cwd()

    if cmd == "version":
        click.echo(f"mcc {_version()}")
        return

    if cmd == "config":
        from mcc_config import load

        config = asyncio.run(load(work_dir))
        click.echo(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

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
