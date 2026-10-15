"""Agent 错误类型层级。

按错误来源分层，方便上层按类型决定重试 / 询问用户 / 直接失败。
"""

from __future__ import annotations


class AgentError(Exception):
    """所有 Agent 错误的基类。"""


class ApiError(AgentError):
    """LLM API 错误（网络 / 鉴权 / 限流 / 服务端 5xx）。"""


class ToolError(AgentError):
    """工具执行失败。"""

    def __init__(self, name: str, msg: str) -> None:
        self.name = name
        self.msg = msg
        super().__init__(f"Tool `{name}` failed: {msg}")


class PermissionDenied(AgentError):
    """权限检查拒绝。"""


class BudgetExceeded(AgentError):
    """预算（USD 或迭代次数）超限。"""


class Cancelled(AgentError):
    """用户取消（Ctrl-C / Esc）。"""
