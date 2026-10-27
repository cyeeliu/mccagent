"""TUI 应用状态与事件应用逻辑。

App 消费 AgentEvent 事件流，维护渲染项列表与流式文本缓冲，
通过 apply_event 更新状态，由渲染层读取状态绘制界面。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from mcc_core import (
    AgentEvent,
    Error,
    LlmUsage,
    Notice,
    PermissionRequest,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)

from mcc_tui.completions import CompletionItem, CompletionProvider

_MAX_ITEMS = 500


class CallStatus(Enum):
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class RenderItem:
    kind: str
    text: str = ""
    call_id: str = ""
    name: str = ""
    args_preview: str = ""
    status: CallStatus = CallStatus.running
    output: str = ""
    is_error: bool = False
    stats: str = ""


@dataclass
class App:
    session_short_id: str
    model_id: str = ""
    context_limit: int = 0
    items: deque[RenderItem] = field(default_factory=lambda: deque(maxlen=_MAX_ITEMS))
    streaming_text: str = ""
    input: str = ""
    cursor: int = 0
    agent_busy: bool = False
    cost_usd: float = 0.0
    scroll: int = 0
    pending_permission: PermissionRequest | None = None
    permission_queue: list[PermissionRequest] = field(default_factory=list)
    permission_feedback: str = ""
    permission_feedback_mode: bool = False
    # token 状态：LlmUsage 事件驱动（轮末权威值），流式期间用估算
    ctx_tokens: int = 0
    turn_in: int = 0
    turn_out: int = 0
    session_in: int = 0
    session_out: int = 0
    cache_read_total: int = 0
    cache_seen_total: int = 0
    # run 起始快照：TurnEnd 时用差值算本轮 run 的 token（多轮 LLM
    # 调用总和），口径与 cost_usd 一致
    _run_start_in: int = 0
    _run_start_out: int = 0
    _run_start_cache: int = 0
    # 补全浮层：输入以 / 开头且无空格时激活
    comp_items: list[CompletionItem] = field(default_factory=list)
    comp_index: int = 0
    comp_dismissed: bool = False

    @property
    def comp_active(self) -> bool:
        return bool(self.comp_items) and not self.comp_dismissed

    @property
    def cache_hit_pct(self) -> float | None:
        """累计缓存命中率；无缓存流量返回 None（显示占位）。"""
        if self.cache_seen_total <= 0:
            return None
        return self.cache_read_total / self.cache_seen_total

    @property
    def est_stream_tokens(self) -> int:
        """流式中的输出 token 估算（4 字符/token 业界惯例）。"""
        return len(self.streaming_text) // 4

    def push(self, item: RenderItem) -> None:
        self.items.append(item)

    def refresh_completions(self, provider: CompletionProvider | None) -> None:
        """输入变化时重算候选；Esc 关闭后本次输入内不再弹。"""
        if self.comp_dismissed or provider is None:
            self.comp_items = []
            return
        self.comp_items = provider(self.input)
        self.comp_index = min(self.comp_index, max(0, len(self.comp_items) - 1))
        if not self.comp_items:
            self.comp_index = 0

    def dismiss_completions(self) -> None:
        """Esc 关闭浮层（直到输入清空或发送后重置）。"""
        self.comp_dismissed = True
        self.comp_items = []

    def apply_completion(self, item: CompletionItem) -> None:
        """补全高亮项到输入框。"""
        self.input = item.insert
        self.cursor = len(self.input)
        self.comp_items = []

    def reset_input(self) -> None:
        """发送/清空输入后的复位。"""
        self.input = ""
        self.cursor = 0
        self.comp_items = []
        self.comp_dismissed = False

    def flush_streaming(self) -> None:
        if self.streaming_text:
            text = self.streaming_text
            self.streaming_text = ""
            self.push(RenderItem(kind="assistant", text=text))

    def resolve_permission(self) -> PermissionRequest | None:
        """结束当前确认（无论用户选了什么），弹出下一个排队的请求。"""
        resolved = self.pending_permission
        self.pending_permission = (
            self.permission_queue.pop(0) if self.permission_queue else None
        )
        self.permission_feedback = ""
        self.permission_feedback_mode = False
        return resolved

    def apply_event(self, ev: AgentEvent) -> None:
        if isinstance(ev, UserEcho):
            self.flush_streaming()
            self.push(RenderItem(kind="user", text=ev.text))
            self._run_start_in = self.session_in
            self._run_start_out = self.session_out
            self._run_start_cache = self.cache_read_total
        elif isinstance(ev, TextDelta):
            self.streaming_text += ev.text
        elif isinstance(ev, ToolCallStart):
            self.flush_streaming()
            self.push(
                RenderItem(
                    kind="tool_call",
                    call_id=ev.id,
                    name=ev.name,
                    args_preview=ev.args_preview,
                    status=CallStatus.running,
                )
            )
        elif isinstance(ev, ToolCallEnd):
            for item in self.items:
                if item.kind == "tool_call" and item.call_id == ev.id:
                    item.status = CallStatus.failed if ev.is_error else CallStatus.done
                    break
            self.push(
                RenderItem(
                    kind="tool_result",
                    call_id=ev.id,
                    output=ev.output,
                    is_error=ev.is_error,
                )
            )
        elif isinstance(ev, LlmUsage):
            # 每轮 LLM 调用完成后的用量快照：上下文占比与缓存命中率的
            # 权威数据源（请求发出时 input 未知，流尾才上报）
            self.turn_in = (
                ev.input_tokens
                + ev.cache_read_input_tokens
                + ev.cache_creation_input_tokens
            )
            self.turn_out = ev.output_tokens
            self.ctx_tokens = self.turn_in
            self.session_in += self.turn_in
            self.session_out += ev.output_tokens
            self.cache_read_total += ev.cache_read_input_tokens
            self.cache_seen_total += self.turn_in
        elif isinstance(ev, TurnEnd):
            self.flush_streaming()
            self.cost_usd = ev.cost_usd
            self.agent_busy = False
            run_in = self.session_in - self._run_start_in
            run_out = self.session_out - self._run_start_out
            run_cache = self.cache_read_total - self._run_start_cache
            if run_in or run_out:
                stats = (
                    f"↳ {run_in:,} in · {run_out:,} out"
                    f" · ${ev.cost_usd:.4f}"
                )
                if run_cache:
                    stats += f" · cache {run_cache:,} hit"
                self.push(RenderItem(kind="turn_stats", stats=stats))
            self.turn_in = 0
            self.turn_out = 0
        elif isinstance(ev, Notice):
            self.push(RenderItem(kind="notice", text=ev.text))
        elif isinstance(ev, Error):
            self.flush_streaming()
            self.push(RenderItem(kind="error", text=ev.text))
            self.agent_busy = False
        elif isinstance(ev, PermissionRequest):
            if self.pending_permission is None:
                self.pending_permission = ev
            else:
                self.permission_queue.append(ev)
