"""Agent 事件流。

AgentLoop 运行时通过 async generator 产出这些事件，
TUI / headless printer 消费它们做流式渲染。
"""

from __future__ import annotations

from dataclasses import dataclass


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
