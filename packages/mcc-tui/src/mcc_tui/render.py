"""TUI 渲染层：基于 Rich 的三栏布局。

消息区（items + 流式文本） / 输入框 / 状态栏，
由 App 状态驱动，不持有可变状态。
消息区高度受限，始终保留输入框与状态栏可见。
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from mcc_tui.app import App, CallStatus, RenderItem

_INPUT_ROWS = 3
_STATUS_ROWS = 1
_PANEL_BORDERS = 2


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


def _count_lines(text: Text) -> int:
    plain = text.plain
    return plain.count("\n") + 1 if plain else 1


def _select_visible(items: list[Text], available: int, scroll: int) -> list[Text]:
    """从末尾往前选取不超过 available 行的 items，支持向上滚动。"""
    if not items:
        return [Text("")]
    end = len(items) - scroll if scroll > 0 else len(items)
    end = max(0, min(end, len(items)))

    visible: list[Text] = []
    line_count = 0
    for text in reversed(items[:end]):
        n = _count_lines(text)
        if line_count + n > available and visible:
            break
        visible.insert(0, text)
        line_count += n
    return visible or [Text("")]


def build_renderable(app: App, console: Console) -> Layout:
    all_items: list[Text] = [_render_item(it) for it in app.items]
    if app.streaming_text:
        all_items.append(_render_item(RenderItem(kind="assistant", text=app.streaming_text)))

    available = max(
        5,
        console.height - _INPUT_ROWS - _STATUS_ROWS - _PANEL_BORDERS,
    )
    visible = _select_visible(all_items, available, app.scroll)

    thinking = " · thinking…" if app.agent_busy else ""
    title = (
        f" mini-claude-code — session {app.session_short_id}"
        f" (${app.cost_usd:.3f}){thinking} "
    )
    msg_panel = Panel(Group(*visible), title=title, border_style="blue")

    cursor_display = app.input[: app.cursor] + "▎" + app.input[app.cursor :]
    input_panel = Panel(Text(cursor_display), title=" input ", border_style="cyan")

    if app.agent_busy:
        status = Text()
        status.append("●", style="yellow")
        status.append(" thinking… (Esc cancel · Ctrl-C quit)")
    else:
        status = Text(" Enter: send · ↑/↓ scroll · Ctrl-C: quit ")

    layout = Layout()
    layout.split_column(
        Layout(msg_panel, name="messages"),
        Layout(input_panel, name="input", size=_INPUT_ROWS),
        Layout(status, name="status", size=_STATUS_ROWS),
    )
    return layout
