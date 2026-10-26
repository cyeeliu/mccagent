"""MCP 多 server 管理器。

连接配置中声明的所有 MCP server，发现的工具适配后注册到 ToolRegistry。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcc_mcp.adapter import McpToolAdapter
from mcc_mcp.client import McpClient, McpError

logger = logging.getLogger(__name__)


class McpClientManager:
    """管理多个 MCP server 连接与工具注册。"""

    def __init__(self) -> None:
        self._clients: dict[str, McpClient] = {}

    @property
    def clients(self) -> dict[str, McpClient]:
        return self._clients

    async def connect_all(
        self,
        servers: dict[str, Any],
    ) -> None:
        """连接所有配置的 server，失败的不中断其他 server。"""
        for name, cfg in servers.items():
            client = McpClient(
                name=name,
                command=cfg.get("command", ""),
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
            )
            try:
                await client.connect()
                self._clients[name] = client
                logger.info(
                    "MCP server %r connected: %d tools",
                    name, len(client.tools),
                )
            except (McpError, Exception) as e:
                logger.warning("MCP server %r failed: %s", name, e)

    def register_all(self, registry: object) -> int:
        """把所有已连接 server 的工具注册到 registry，返回注册数量。"""
        count = 0
        for server_name, client in self._clients.items():
            for tool_info in client.tools:
                try:
                    adapter = McpToolAdapter(server_name, tool_info, client)
                    registry.register(adapter)
                    count += 1
                except (ValueError, Exception) as e:
                    logger.warning(
                        "MCP tool %r/%r skipped: %s",
                        server_name, tool_info.get("name"), e,
                    )
        return count

    async def close_all(self) -> None:
        """关闭所有 server 子进程。"""
        await asyncio.gather(
            *(c.close() for c in self._clients.values()),
            return_exceptions=True,
        )
        self._clients.clear()
