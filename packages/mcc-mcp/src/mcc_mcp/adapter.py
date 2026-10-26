"""MCP 工具适配器：把 MCP server 暴露的工具适配成 mcc Tool 协议。

工具名前缀 ``mcp__<server>__<tool>`` 防冲突，与 Claude Code 命名兼容。
"""

from __future__ import annotations

import re
from typing import Any

from mcc_core import ToolContext, ToolOutput

_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    """替换模型工具名字母表之外的字符为下划线。"""
    normalized = _DISALLOWED.sub("_", name)
    if not normalized:
        raise ValueError(f"MCP name {name!r} normalizes to empty string")
    return normalized


class McpToolAdapter:
    """把单个 MCP 工具适配成 mcc Tool 协议。

    execute 时通过 McpClient.call_tool 转发给 server 子进程。
    """

    def __init__(
        self,
        server_name: str,
        tool_info: dict[str, Any],
        client: object,
    ) -> None:
        raw_name = str(tool_info.get("name", ""))
        if not raw_name:
            raise ValueError("MCP tool info missing 'name'")
        self._raw_name = raw_name
        self._server_name = server_name
        self._name = (
            f"mcp__{normalize_mcp_name(server_name)}"
            f"__{normalize_mcp_name(raw_name)}"
        )
        if len(self._name) > 64:
            raise ValueError(
                f"MCP tool name too long (>64): {self._name}"
            )
        self._description = str(tool_info.get("description", ""))
        self._input_schema = tool_info.get("inputSchema", {})
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def input_schema(self) -> dict[str, Any]:
        return self._input_schema

    async def execute(
        self, input: dict[str, Any], ctx: ToolContext
    ) -> ToolOutput:
        text, is_error = await self._client.call_tool(self._raw_name, input)
        return ToolOutput(content=text, is_error=is_error)
