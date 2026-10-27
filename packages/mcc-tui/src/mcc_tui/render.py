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
_STATUS_ROWS = 3
_PANEL_BORDERS = 2
_TEXT_INDENT = 2
_COMP_MAX_ROWS = 7


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


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
    elif item.kind == "turn_stats":
        text.append(f"  {item.stats}", style="dim italic")
    elif item.kind == "error":
        text.append(f"✗ error: {item.text}", style="bold red")
    return text


def _extract_line(text: Text, start: int, end: int) -> Text:
    """提取 text.plain[start:end] 并复制落在该范围内的 spans，保留样式。"""
    plain = text.plain[start:end]
    line = Text(plain)
    for span in text.spans:
        if span.end <= start or span.start >= end:
            continue
        s = max(0, span.start - start)
        e = min(end - start, span.end - start)
        line.stylize(span.style, s, e)
    return line


def _wrap_line(text: Text, start: int, end: int, width: int) -> list[Text]:
    """把 [start, end) 范围按 width 拆成多行 Text。"""
    line_len = end - start
    if line_len <= width:
        return [_extract_line(text, start, end)]
    result: list[Text] = []
    pos = start
    while pos < end:
        chunk_end = min(pos + width, end)
        result.append(_extract_line(text, pos, chunk_end))
        pos = chunk_end
    return result


def _split_lines(text: Text, width: int = 80) -> list[Text]:
    """把多行 Text 拆成单行 Text 列表，处理 word wrap，保留样式。"""
    plain = text.plain
    if not plain:
        return [Text("")]
    result: list[Text] = []
    line_start = 0
    for i, ch in enumerate(plain):
        if ch == "\n":
            result.extend(_wrap_line(text, line_start, i, width))
            line_start = i + 1
    if line_start < len(plain):
        result.extend(_wrap_line(text, line_start, len(plain), width))
    return result or [Text("")]


def _select_visible(
    items: list[Text], available: int, scroll: int, width: int = 80
) -> list[Text]:
    """行级滑动窗口：把所有 item 展开为行，按 scroll 裁取 available 行。"""
    if not items:
        return [Text("")]

    all_lines: list[Text] = []
    for item in items:
        all_lines.extend(_split_lines(item, width))

    total = len(all_lines)
    max_scroll = max(0, total - available)
    scroll = min(max(0, scroll), max_scroll)

    bottom = total - scroll
    top = max(0, bottom - available)

    return all_lines[top:bottom] or [Text("")]


def _max_scroll(items: list[Text], available: int, width: int = 80) -> int:
    if not items:
        return 0
    total = sum(len(_split_lines(t, width)) for t in items)
    return max(0, total - available)


def _completion_panel(app: App) -> Panel | None:
    """slash 补全浮层：候选项列表，高亮行反色。"""
    if not app.comp_active:
        return None
    text = Text()
    for idx, item in enumerate(app.comp_items[:_COMP_MAX_ROWS]):
        style = "reverse" if idx == app.comp_index else ""
        line = item.label
        if item.hint:
            line += f"  {_truncate(item.hint, 40)}"
        text.append(line + "\n", style=style)
    return Panel(text, title=" commands ", border_style="magenta")


def _status_info(app: App) -> Text:
    """常驻信息行：模型 · 上下文占比（颜色分级）· 缓存命中率 · 成本。"""
    text = Text()
    text.append(f" {app.model_id or '—'}", style="bold")
    if app.context_limit > 0:
        pct = app.ctx_tokens / app.context_limit
        style = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "red"
        text.append(
            f" · ctx {_fmt_tokens(app.ctx_tokens)}/{_fmt_tokens(app.context_limit)}"
            f" {pct:.0%}",
            style=style,
        )
        if pct >= 0.85:
            text.append(" ! near limit", style="bold red")
    else:
        text.append(f" · ctx {_fmt_tokens(app.ctx_tokens)}", style="dim")
    cache = app.cache_hit_pct
    text.append(
        f" · cache {cache:.0%}" if cache is not None else " · cache —",
        style="dim" if cache is None else "cyan",
    )
    text.append(f" · ${app.cost_usd:.4f}", style="dim")
    return text


def _status_hint(app: App) -> Text:
    """下行提示：feedback/permission/busy/idle 四态。"""
    if app.permission_feedback_mode:
        return Text(" Enter: submit · Esc: back to choice ")
    if app.pending_permission is not None:
        status = Text()
        status.append("⚠ ", style="bold yellow")
        status.append("y: allow · a: always · n: deny · e: deny with comment")
        return status
    if app.agent_busy:
        status = Text()
        status.append("●", style="yellow")
        status.append(" thinking… (Esc cancel · Ctrl-C quit)")
        return status
    if app.comp_active:
        return Text(" Tab/Enter: complete · ↑/↓: choose · Esc: close ")
    return Text(" Enter: send · /: commands · ↑/↓: scroll · Ctrl-C: quit ")


def build_renderable(app: App, console: Console) -> Layout:
    all_items: list[Text] = [_render_item(it) for it in app.items]
    if app.streaming_text:
        all_items.append(_render_item(RenderItem(kind="assistant", text=app.streaming_text)))

    comp_panel = _completion_panel(app)
    comp_rows = (len(app.comp_items[:_COMP_MAX_ROWS]) + _PANEL_BORDERS) if comp_panel else 0

    available = max(
        5,
        console.height - _INPUT_ROWS - _STATUS_ROWS - _PANEL_BORDERS - comp_rows,
    )
    text_width = max(20, console.width - _PANEL_BORDERS - _TEXT_INDENT)
    app.scroll = min(app.scroll, _max_scroll(all_items, available, text_width))
    visible = _select_visible(all_items, available, app.scroll, text_width)

    thinking = (
        f" · thinking… ~{_fmt_tokens(app.est_stream_tokens)} tok"
        if app.agent_busy
        else ""
    )
    title = (
        f" mini-claude-code — session {app.session_short_id}"
        f" (${app.cost_usd:.3f}){thinking} "
    )
    msg_panel = Panel(Group(*visible), title=title, border_style="blue")

    if app.permission_feedback_mode:
        cursor_display = app.permission_feedback + "▎"
        input_panel = Panel(
            Text(cursor_display),
            title=" feedback — denied + attach your comment ",
            border_style="yellow",
        )
    elif app.pending_permission is not None:
        p = app.pending_permission
        prompt = Text()
        prompt.append("⚠ ", style="bold yellow")
        prompt.append(p.message, style="bold")
        prompt.append("   ")
        prompt.append("[y]", style="bold green")
        prompt.append(" allow  ")
        prompt.append("[a]", style="bold green")
        prompt.append(" always  ")
        prompt.append("[n]", style="bold red")
        prompt.append(" deny  ")
        prompt.append("[e]", style="bold cyan")
        prompt.append(" deny+comment")
        input_panel = Panel(
            prompt, title=" permission required ", border_style="yellow"
        )
    else:
        cursor_display = app.input[: app.cursor] + "▎" + app.input[app.cursor :]
        input_panel = Panel(Text(cursor_display), title=" input ", border_style="cyan")

    status = Group(_status_info(app), _status_hint(app))

    rows: list[Layout] = [Layout(msg_panel, name="messages")]
    if comp_panel is not None:
        rows.append(Layout(comp_panel, name="completions", size=comp_rows))
    rows.append(Layout(input_panel, name="input", size=_INPUT_ROWS))
    rows.append(Layout(status, name="status", size=_STATUS_ROWS))

    layout = Layout()
    layout.split_column(*rows)
    return layout
