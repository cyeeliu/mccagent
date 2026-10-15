"""终端 UI 入口。

基于 Rich 的三栏布局终端界面，消费 AgentEvent 事件流做流式渲染。
UI 与业务解耦：用户输入通过队列上抛，Agent 事件通过队列下发。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mcc_core import AgentEvent
from rich.console import Console
from rich.live import Live

from mcc_tui.app import App
from mcc_tui.keys import Key, KeyReader
from mcc_tui.render import build_renderable


@dataclass
class TuiHandles:
    events: asyncio.Queue[AgentEvent]
    input_tx: asyncio.Queue[str]
    session_short_id: str


async def run_tui(handles: TuiHandles, console: Console | None = None) -> None:
    """运行 TUI 事件循环，直到用户退出或输入结束。"""
    console = console or Console()
    app = App(session_short_id=handles.session_short_id)

    key_queue: asyncio.Queue = asyncio.Queue()

    with KeyReader() as reader:
        reader_task = asyncio.create_task(_key_reader_loop(reader, key_queue))
        renderable = build_renderable(app, console)
        with Live(renderable, auto_refresh=False, console=console, screen=False) as live:
            try:
                while True:
                    live.update(build_renderable(app, console), refresh=True)
                    if await _step(app, handles, key_queue):
                        break
            finally:
                reader_task.cancel()


async def _key_reader_loop(reader: KeyReader, queue: asyncio.Queue) -> None:
    """持续读取按键放入队列，避免 read_key 被 cancel 后丢失字符。"""
    while True:
        key, _ = await reader.read_key()
        await queue.put(key)


async def _step(app: App, handles: TuiHandles, key_queue: asyncio.Queue) -> bool:
    key_task = asyncio.create_task(key_queue.get())
    ev_task = asyncio.create_task(handles.events.get())
    done, _pending = await asyncio.wait(
        {key_task, ev_task}, return_when=asyncio.FIRST_COMPLETED
    )
    should_quit = False
    if key_task in done:
        ev_task.cancel()
        key = key_task.result()
        should_quit = _handle_key(app, handles, key)
    if ev_task in done:
        key_task.cancel()
        try:
            ev = ev_task.result()
            app.apply_event(ev)
        except asyncio.CancelledError:
            pass
    return should_quit


def _handle_key(app: App, handles: TuiHandles, key: Key | str) -> bool:
    if key in (Key.ctrl_c, Key.ctrl_d):
        return True
    if key == Key.escape and app.agent_busy:
        app.push(_notice_item("(cancel requested)"))
        return False
    if key == Key.enter and not app.agent_busy:
        text = app.input
        app.input = ""
        app.cursor = 0
        if text.strip():
            app.agent_busy = True
            handles.input_tx.put_nowait(text)
        return False
    if key == Key.backspace:
        if app.cursor > 0:
            app.input = app.input[: app.cursor - 1] + app.input[app.cursor :]
            app.cursor -= 1
        return False
    if key == Key.left:
        app.cursor = max(0, app.cursor - 1)
        return False
    if key == Key.right:
        app.cursor = min(len(app.input), app.cursor + 1)
        return False
    if key == Key.home:
        app.cursor = 0
        return False
    if key == Key.end:
        app.cursor = len(app.input)
        return False
    if key == Key.page_up:
        app.scroll += 5
        return False
    if key == Key.page_down:
        app.scroll = max(0, app.scroll - 5)
        return False
    if isinstance(key, str):
        app.input = app.input[: app.cursor] + key + app.input[app.cursor :]
        app.cursor += 1
    return False


def _notice_item(text: str) -> Any:
    from mcc_tui.app import RenderItem

    return RenderItem(kind="notice", text=text)


__all__ = ["App", "TuiHandles", "run_tui"]
__version__ = "0.1.0"
