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
class ModelChanged:
    """主对话模型切换事件（/model 或 /config set model.main 触发）。

    TUI 据此更新状态栏模型 ID；AgentLoop.model 已由 slash handler
    即刻改写，本事件仅用于通知前端同步展示。
    """

    model_id: str


@dataclass
class SessionChanged:
    """会话切换事件（/resume 恢复历史会话时触发）。

    TUI 据此更新状态栏 session id；agent.history 已由 slash handler
    即刻改写，本事件仅用于通知前端同步展示。
    """

    session_id: str


@dataclass
class GoalStatus:
    """Goal Loop 状态变更事件。

    action 取值：set / cleared / status / achieved / failed /
    block / limit / error。TUI 据此渲染目标面板。
    """

    action: str
    condition: str
    reason: str
    iterations: int = 0


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


@dataclass
class SelectionOption:
    """一个可选项：label 供浮层展示，value 回传给事件源。"""

    label: str
    value: str


@dataclass
class SelectionRequest:
    """请求用户从候选列表中选择一项（AgentLoop → TUI）。

    用于 /resume 等需要交互式选择的 slash 命令：事件源发出请求
    后挂起，TUI 渲染浮层并回传 SelectionDecision。
    """

    id: str
    title: str
    options: list[SelectionOption]


@dataclass
class SelectionDecision:
    """用户对选择请求的决策（TUI → AgentLoop 回传）。

    value 为 None 表示用户按 Esc 取消。
    """

    request_id: str
    value: str | None


AgentEvent = (
    UserEcho
    | TextDelta
    | ToolCallStart
    | ToolCallEnd
    | LlmUsage
    | TurnEnd
    | Notice
    | ModelChanged
    | SessionChanged
    | GoalStatus
    | Error
    | PermissionRequest
    | SelectionRequest
)
