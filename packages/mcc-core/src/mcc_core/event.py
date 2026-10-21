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
class LlmUsage:
    """一次 LLM 调用完成后的用量快照（流结束时才能拿到）。

    请求发出时 input token 数未知（Anthropic 在 message_start、
    OpenAI 在流尾 usage chunk 才上报），因此用量在每轮流完后
    单独发射——事件流上 token 统计的权威来源。
    """

    iteration: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0


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
    | LlmUsage
    | TurnEnd
    | Notice
    | Error
    | PermissionRequest
)
