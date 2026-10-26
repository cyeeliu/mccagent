"""MCP stdio client：spawn server 子进程，JSON-RPC 2.0 通信。

协议流程：initialize 握手 → tools/list 发现 → tools/call 转发。
每行一个 JSON-RPC 消息（newline-delimited）。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"
CLIENT_INFO = {"name": "mcc", "version": "0.1.0"}


class McpError(Exception):
    """MCP 协议或传输层错误。"""


class McpClient:
    """单个 MCP server 的 stdio 连接。"""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self._proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._tools: list[dict[str, Any]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def connect(self) -> None:
        """spawn server 子进程，initialize 握手，发现工具。"""
        full_env = {**os.environ, **self.env}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=full_env,
            )
        except FileNotFoundError as e:
            raise McpError(
                f"MCP server {self.name!r}: command {self.command!r} not found"
            ) from e
        except OSError as e:
            raise McpError(
                f"MCP server {self.name!r}: failed to start: {e}"
            ) from e

        await self._initialize()
        self._tools = await self._tools_list()

    async def _initialize(self) -> dict[str, Any]:
        result = await self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        await self._notify("notifications/initialized")
        return result

    async def _tools_list(self) -> list[dict[str, Any]]:
        result = await self._rpc("tools/list", {})
        return result.get("tools", [])

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[str, bool]:
        """调用 server 工具，返回 (text, is_error)。"""
        result = await self._rpc("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        is_error = bool(result.get("isError", False))
        return text, is_error

    async def close(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """发 JSON-RPC request，等对应 id 的 response。"""
        if self._proc is None or self._proc.stdout is None:
            raise McpError(f"server {self.name!r} not connected")
        self._id += 1
        req_id = self._id
        line = json.dumps({
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "method": method,
            "params": params,
        }) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()
        return await self._read_response(req_id)

    async def _notify(self, method: str) -> None:
        """发 JSON-RPC notification（无 id，不期待响应）。"""
        if self._proc is None:
            raise McpError(f"server {self.name!r} not connected")
        line = json.dumps({
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
        }) + "\n"
        self._proc.stdin.write(line.encode())
        await self._proc.stdin.drain()

    async def _read_response(self, expected_id: int) -> Any:
        if self._proc is None or self._proc.stdout is None:
            raise McpError(f"server {self.name!r} not connected")
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                raise McpError(
                    f"server {self.name!r} closed stdout before response"
                )
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                raise McpError(
                    f"server {self.name!r}: invalid JSON: {e}"
                ) from e
            if "id" not in msg:
                continue
            if msg["id"] != expected_id:
                continue
            if "error" in msg and msg["error"] is not None:
                err = msg["error"]
                raise McpError(
                    f"server {self.name!r} JSON-RPC error: "
                    f"{err.get('message', err)}"
                )
            return msg.get("result")
