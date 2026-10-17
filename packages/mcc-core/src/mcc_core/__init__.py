"""核心类型层（无外部 I/O 依赖）。

是整个依赖图的最底层，提供消息、内容块、工具协议、事件与错误类型。
"""

from mcc_core.error import (
    AgentError,
    ApiError,
    BudgetExceeded,
    Cancelled,
    PermissionDenied,
    ToolError,
)
from mcc_core.event import (
    AgentEvent,
    Error,
    Notice,
    PermissionDecision,
    PermissionRequest,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_core.tool import Tool, ToolCall, ToolContext, ToolOutput
from mcc_core.types import (
    CacheControl,
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "AgentError",
    "ApiError",
    "BudgetExceeded",
    "Cancelled",
    "PermissionDenied",
    "ToolError",
    "AgentEvent",
    "Error",
    "Notice",
    "PermissionDecision",
    "PermissionRequest",
    "TextDelta",
    "ToolCallEnd",
    "ToolCallStart",
    "TurnEnd",
    "UserEcho",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolOutput",
    "CacheControl",
    "ContentBlock",
    "Message",
    "Role",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
]

__version__ = "0.1.0"
