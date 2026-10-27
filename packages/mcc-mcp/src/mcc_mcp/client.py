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
DEFAULT_REQUEST_TIMEOUT_SECS = 60.0


class McpError(Exception):
    """MCP 协议或传输层错误。"""


class McpClient:
    """单个 MCP server 的 stdio 连接。

    所有响应由唯一 reader task 消费，再按 JSON-RPC id 分发给等待中的请求。
    这样并发工具调用不会互相消费并丢弃对方的响应。
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        request_timeout_secs: float = DEFAULT_REQUEST_TIMEOUT_SECS,
    ) -> None:
        if request_timeout_secs <= 0:
            raise ValueError("request_timeout_secs must be positive")
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.request_timeout_secs = request_timeout_secs
        self._proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._tools: list[dict[str, Any]] = []
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()

    @property
    def tools(self) -> list[dict[str, Any]]:
        return self._tools

    async def connect(self) -> None:
        """spawn server 子进程，启动唯一响应读取器，initialize 握手并发现工具。"""
        if self._proc is not None:
            raise McpError(f"server {self.name!r} is already connected")

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

        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"mcp-reader-{self.name}"
        )
        try:
            await self._initialize()
            self._tools = await self._tools_list()
        except BaseException:
            await self.close()
            raise

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
        proc = self._proc
        self._proc = None

        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except TimeoutError:
                proc.kill()
                await proc.wait()

        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None and not reader_task.done():
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass

        self._fail_pending(McpError(f"server {self.name!r} connection closed"))

    async def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        """发送 JSON-RPC request，并等待 reader 按 id 分发对应响应。"""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise McpError(f"server {self.name!r} not connected")
        if proc.returncode is not None:
            raise McpError(f"server {self.name!r} process has exited")

        self._id += 1
        req_id = self._id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[req_id] = future
        line = json.dumps({
            "jsonrpc": JSONRPC_VERSION,
            "id": req_id,
            "method": method,
            "params": params,
        }) + "\n"

        try:
            async with self._write_lock:
                if self._proc is not proc or proc.returncode is not None:
                    raise McpError(f"server {self.name!r} is no longer connected")
                proc.stdin.write(line.encode())
                await proc.stdin.drain()
            return await asyncio.wait_for(future, timeout=self.request_timeout_secs)
        except TimeoutError as e:
            raise McpError(
                f"server {self.name!r} request {method!r} timed out "
                f"after {self.request_timeout_secs:g}s"
            ) from e
        except (BrokenPipeError, ConnectionError) as e:
            raise McpError(f"server {self.name!r} write failed: {e}") from e
        finally:
            self._pending.pop(req_id, None)
            if not future.done():
                future.cancel()

    async def _notify(self, method: str) -> None:
        """发送 JSON-RPC notification（无 id，不期待响应）。"""
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise McpError(f"server {self.name!r} not connected")
        line = json.dumps({"jsonrpc": JSONRPC_VERSION, "method": method}) + "\n"
        try:
            async with self._write_lock:
                if self._proc is not proc or proc.returncode is not None:
                    raise McpError(f"server {self.name!r} is no longer connected")
                proc.stdin.write(line.encode())
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionError) as e:
            raise McpError(
                f"server {self.name!r} notification write failed: {e}"
            ) from e

    async def _read_loop(self) -> None:
        """唯一的 stdout reader：按 response id 将结果分发给对应 Future。"""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    raise McpError(f"server {self.name!r} closed stdout before response")
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise McpError(f"server {self.name!r}: invalid JSON: {e}") from e

                if "id" not in msg:
                    continue
                future = self._pending.pop(msg["id"], None)
                if future is None or future.done():
                    continue  # 超时或取消的请求可能仍会收到迟到响应
                if msg.get("error") is not None:
                    err = msg["error"]
                    future.set_exception(McpError(
                        f"server {self.name!r} JSON-RPC error: "
                        f"{err.get('message', err)}"
                    ))
                else:
                    future.set_result(msg.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error = e if isinstance(e, McpError) else McpError(
                f"server {self.name!r} response reader failed: {e}"
            )
            self._fail_pending(error)

    def _fail_pending(self, error: McpError) -> None:
        """传输不可用时，统一失败所有进行中的请求。"""
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)
