"""Agent 事件流。

AgentLoop 运行时通过 async generator 产出这些事件，
TUI / headless printer 消费它们做流式渲染。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserEcho:
    text: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallStart:
    id: str
    name: str
    args_preview: str


@dataclass
class ToolCallEnd:
    id: str
    output: str
    is_error: bool


@dataclass
class TurnEnd:
    cost_usd: float


@dataclass
class Notice:
    text: str


@dataclass
class Error:
    text: str


@dataclass
class PermissionRequest:
    id: int
    message: str
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionDecision:
    """用户对权限请求的决策（TUI → AgentLoop 回传）。"""

    request_id: int
    action: str  # "allow" | "allow_always" | "deny" | "feedback"
    feedback: str = ""


AgentEvent = (
    UserEcho
    | TextDelta
    | ToolCallStart
    | ToolCallEnd
    | TurnEnd
    | Notice
    | Error
    | PermissionRequest
)
