"""终端 UI 入口。

基于 Rich 的三栏布局终端界面，消费 AgentEvent 事件流做流式渲染。
UI 与业务解耦：用户输入通过队列上抛，Agent 事件通过队列下发。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from mcc_core import AgentEvent, PermissionDecision
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
    permission_tx: asyncio.Queue[PermissionDecision] | None = None
    completions: Any = None  # CompletionProvider：slash 补全候选
    model_id: str = ""
    context_limit: int = 0


async def run_tui(handles: TuiHandles, console: Console | None = None) -> None:
    """运行 TUI 事件循环，直到用户退出或输入结束。"""
    console = console or Console()
    app = App(
        session_short_id=handles.session_short_id,
        model_id=handles.model_id,
        context_limit=handles.context_limit,
    )

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
    if app.permission_feedback_mode:
        return _handle_feedback_key(app, handles, key)
    if app.pending_permission is not None:
        return _handle_permission_key(app, handles, key)

    # 补全浮层激活时的键位优先：↑↓ 选择、Tab 补全、Enter 补全选中
    # 项（已是精确匹配则照常发送）、Esc 关闭浮层
    if app.comp_active:
        if key == Key.up:
            app.comp_index = (app.comp_index - 1) % len(app.comp_items)
            return False
        if key == Key.down:
            app.comp_index = (app.comp_index + 1) % len(app.comp_items)
            return False
        if key == Key.tab:
            app.apply_completion(app.comp_items[app.comp_index])
            return False
        if key == Key.escape:
            app.dismiss_completions()
            return False
        if key == Key.enter and not app.agent_busy:
            current = app.comp_items[app.comp_index]
            if app.input.strip() == current.insert:
                app.dismiss_completions()
                return _send_input(app, handles)
            app.apply_completion(current)
            return False

    if key == Key.escape and app.agent_busy:
        app.push(_notice_item("(cancel requested)"))
        return False
    if key == Key.enter and not app.agent_busy:
        return _send_input(app, handles)
    if key == Key.backspace:
        if app.cursor > 0:
            app.input = app.input[: app.cursor - 1] + app.input[app.cursor :]
            app.cursor -= 1
        app.refresh_completions(handles.completions)
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
    if key == Key.up:
        app.scroll += 2
        return False
    if key == Key.down:
        app.scroll = max(0, app.scroll - 2)
        return False
    if key == Key.page_up:
        app.scroll += 10
        return False
    if key == Key.page_down:
        app.scroll = max(0, app.scroll - 10)
        return False
    if isinstance(key, str):
        app.input = app.input[: app.cursor] + key + app.input[app.cursor :]
        app.cursor += 1
        app.refresh_completions(handles.completions)
    return False


def _send_input(app: App, handles: TuiHandles) -> bool:
    """Enter 发送：清空输入、关闭补全、非空则进 agent。"""
    text = app.input
    app.reset_input()
    if text.strip():
        app.agent_busy = True
        handles.input_tx.put_nowait(text)
    return False


def _send_decision(
    app: App, handles: TuiHandles, action: str, feedback: str = ""
) -> None:
    """向 agent 回传当前权限请求的决策，并弹出下一个排队请求。"""
    req = app.resolve_permission()
    if req is not None and handles.permission_tx is not None:
        handles.permission_tx.put_nowait(
            PermissionDecision(
                request_id=req.id, action=action, feedback=feedback
            )
        )
    app.push(
        _notice_item(
            f"⚠ permission {'allowed' if action in ('allow', 'allow_always') else 'denied'}"
            + (" (always)" if action == "allow_always" else "")
            + (f": {feedback}" if feedback else "")
        )
    )


def _handle_permission_key(app: App, handles: TuiHandles, key: Key | str) -> bool:
    if key == "y":
        _send_decision(app, handles, "allow")
    elif key == "a":
        _send_decision(app, handles, "allow_always")
    elif key == "n" or key == Key.escape:
        _send_decision(app, handles, "deny")
    elif key == "e":
        app.permission_feedback_mode = True
        app.permission_feedback = ""
    return False


def _handle_feedback_key(app: App, handles: TuiHandles, key: Key | str) -> bool:
    if key == Key.enter:
        feedback = app.permission_feedback.strip()
        _send_decision(app, handles, "feedback", feedback)
    elif key == Key.escape:
        app.permission_feedback_mode = False
        app.permission_feedback = ""
    elif key == Key.backspace:
        app.permission_feedback = app.permission_feedback[:-1]
    elif isinstance(key, str):
        app.permission_feedback += key
    return False


def _notice_item(text: str) -> Any:
    from mcc_tui.app import RenderItem

    return RenderItem(kind="notice", text=text)


__all__ = ["App", "TuiHandles", "run_tui"]
__version__ = "0.1.0"
