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
    Notice,
    PermissionRequest,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)

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


@dataclass
class App:
    session_short_id: str
    items: deque[RenderItem] = field(default_factory=lambda: deque(maxlen=_MAX_ITEMS))
    streaming_text: str = ""
    input: str = ""
    cursor: int = 0
    agent_busy: bool = False
    cost_usd: float = 0.0
    scroll: int = 0

    def push(self, item: RenderItem) -> None:
        self.items.append(item)

    def flush_streaming(self) -> None:
        if self.streaming_text:
            text = self.streaming_text
            self.streaming_text = ""
            self.push(RenderItem(kind="assistant", text=text))

    def apply_event(self, ev: AgentEvent) -> None:
        if isinstance(ev, UserEcho):
            self.flush_streaming()
            self.push(RenderItem(kind="user", text=ev.text))
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
        elif isinstance(ev, TurnEnd):
            self.flush_streaming()
            self.cost_usd = ev.cost_usd
            self.agent_busy = False
        elif isinstance(ev, Notice):
            self.push(RenderItem(kind="notice", text=ev.text))
        elif isinstance(ev, Error):
            self.flush_streaming()
            self.push(RenderItem(kind="error", text=ev.text))
            self.agent_busy = False
        elif isinstance(ev, PermissionRequest):
            self.push(RenderItem(kind="notice", text=f"⚠ permission: {ev.message}"))
