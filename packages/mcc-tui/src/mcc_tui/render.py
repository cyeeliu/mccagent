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
    """行级滑动窗口：scroll 为从底部往上的行偏移，始终显示 available 行内容。"""
    if not items:
        return [Text("")]

    item_lines = [(text, _count_lines(text)) for text in items]
    total_lines = sum(n for _, n in item_lines)

    max_scroll = max(0, total_lines - available)
    scroll = min(max(0, scroll), max_scroll)

    bottom_line = total_lines - scroll
    top_line = max(0, bottom_line - available)

    visible: list[Text] = []
    current = 0
    for text, n in item_lines:
        if current + n > top_line and current < bottom_line:
            visible.append(text)
        current += n

    return visible or [Text("")]


def _max_scroll(items: list[Text], available: int) -> int:
    if not items:
        return 0
    total = sum(_count_lines(t) for t in items)
    return max(0, total - available)


def build_renderable(app: App, console: Console) -> Layout:
    all_items: list[Text] = [_render_item(it) for it in app.items]
    if app.streaming_text:
        all_items.append(_render_item(RenderItem(kind="assistant", text=app.streaming_text)))

    available = max(
        5,
        console.height - _INPUT_ROWS - _STATUS_ROWS - _PANEL_BORDERS,
    )
    app.scroll = min(app.scroll, _max_scroll(all_items, available))
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
