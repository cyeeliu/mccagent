"""TUI 渲染层：基于 Rich 的三栏布局。

消息区（items + 流式文本） / 输入框 / 状态栏，
由 App 状态驱动，不持有可变状态。
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from mcc_tui.app import App, CallStatus, RenderItem


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _render_item(item: RenderItem) -> Text:
    text = Text()
    if item.kind == "user":
        text.append("> ", style="bold cyan")
        text.append(item.text)
    elif item.kind == "assistant":
        text.append("◎ assistant", style="bold green")
        text.append("\n")
        for line in item.text.splitlines() or [""]:
            text.append(f"  {line}\n")
    elif item.kind == "tool_call":
        icon, color = {
            CallStatus.running: ("▶", "yellow"),
            CallStatus.done: ("✓", "green"),
            CallStatus.failed: ("✗", "red"),
        }[item.status]
        text.append(f" {icon} ", style=color)
        text.append(item.name, style="bold")
        text.append(f"({_truncate(item.args_preview, 80)})")
    elif item.kind == "tool_result":
        style = "red" if item.is_error else "dim"
        lines = item.output.splitlines()[:8] or [""]
        for line in lines:
            text.append(f"    {line}\n", style=style)
    elif item.kind == "notice":
        text.append(f"· {item.text}", style="blue")
    elif item.kind == "error":
        text.append(f"✗ error: {item.text}", style="bold red")
    return text


def build_renderable(app: App, console: Console) -> Group:
    msg_lines: list[Text] = []
    for item in app.items:
        msg_lines.append(_render_item(item))
    if app.streaming_text:
        stream_item = RenderItem(kind="assistant", text=app.streaming_text)
        msg_lines.append(_render_item(stream_item))

    if not msg_lines:
        msg_lines.append(Text(""))

    thinking = " · thinking…" if app.agent_busy else ""
    title = (
        f" mini-claude-code — session {app.session_short_id}"
        f" (${app.cost_usd:.3f}){thinking} "
    )
    msg_group = Group(*msg_lines)
    msg_panel = Panel(msg_group, title=title, border_style="blue")

    cursor_display = app.input[: app.cursor] + "▎" + app.input[app.cursor :]
    input_panel = Panel(Text(cursor_display), title=" input ", border_style="cyan")

    if app.agent_busy:
        status = Text()
        status.append("●", style="yellow")
        status.append(" thinking… (Esc cancel · Ctrl-C quit)")
    else:
        status = Text(" Enter: send · PgUp/PgDn: scroll · Ctrl-C: quit ")

    return Group(msg_panel, input_panel, status)
