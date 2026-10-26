"""MCP (Model Context Protocol) client。

通过 stdio JSON-RPC 2.0 连接外部 MCP server，发现工具并适配成
mcc 的 Tool 协议注册到 ToolRegistry，AgentLoop 透明使用。
"""

from __future__ import annotations

from mcc_mcp.adapter import McpToolAdapter, normalize_mcp_name
from mcc_mcp.client import McpClient, McpError
from mcc_mcp.manager import McpClientManager

__all__ = [
    "McpClient",
    "McpClientManager",
    "McpError",
    "McpToolAdapter",
    "normalize_mcp_name",
]

__version__ = "0.1.0"
