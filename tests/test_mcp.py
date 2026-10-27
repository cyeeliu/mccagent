"""MCP 测试：adapter、client（mock server）、manager。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from mcc_core import ToolContext
from mcc_mcp import McpClient, McpClientManager, McpError, McpToolAdapter, normalize_mcp_name
from mcc_tools import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "mock_mcp_server.py"


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


class _MockClient:
    """记录 call_tool 调用的 mock client。"""

    def __init__(self, return_text: str = "ok", return_error: bool = False) -> None:
        self._return_text = return_text
        self._return_error = return_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        self.calls.append((name, arguments))
        return self._return_text, self._return_error


class TestNormalizeName:
    def test_plain_name_unchanged(self) -> None:
        assert normalize_mcp_name("echo") == "echo"

    def test_name_with_dot_and_slash(self) -> None:
        assert normalize_mcp_name("fs.read/file") == "fs_read_file"

    def test_name_with_spaces(self) -> None:
        assert normalize_mcp_name("my tool") == "my_tool"

    def test_all_special_chars_become_underscores(self) -> None:
        assert normalize_mcp_name("!!!") == "___"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_mcp_name("")


class TestMcpToolAdapter:
    def test_name_prefix(self) -> None:
        adapter = McpToolAdapter(
            "myserver",
            {"name": "echo", "description": "d", "inputSchema": {}},
            _MockClient(),
        )
        assert adapter.name == "mcp__myserver__echo"
        assert adapter.description == "d"
        assert adapter.input_schema() == {}

    def test_name_with_special_chars_normalized(self) -> None:
        adapter = McpToolAdapter(
            "fs.server",
            {"name": "read/file", "description": "", "inputSchema": {}},
            _MockClient(),
        )
        assert adapter.name == "mcp__fs_server__read_file"

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValueError):
            McpToolAdapter("srv", {"description": "d"}, _MockClient())

    def test_name_too_long_raises(self) -> None:
        long_name = "a" * 70
        with pytest.raises(ValueError):
            McpToolAdapter("srv", {"name": long_name}, _MockClient())

    @pytest.mark.asyncio
    async def test_execute_forwards_to_client(self, ctx: ToolContext) -> None:
        client = _MockClient(return_text="hello", return_error=False)
        adapter = McpToolAdapter(
            "srv", {"name": "echo", "description": "", "inputSchema": {}}, client
        )
        out = await adapter.execute({"message": "hello"}, ctx)
        assert out.content == "hello"
        assert out.is_error is False
        assert client.calls == [("echo", {"message": "hello"})]

    @pytest.mark.asyncio
    async def test_execute_propagates_error_flag(self, ctx: ToolContext) -> None:
        client = _MockClient(return_text="boom", return_error=True)
        adapter = McpToolAdapter(
            "srv", {"name": "fail", "description": "", "inputSchema": {}}, client
        )
        out = await adapter.execute({"message": "boom"}, ctx)
        assert out.content == "boom"
        assert out.is_error is True


class TestMcpClientRealServer:
    """用 mock_mcp_server.py 子进程测试真实 stdio 通信。"""

    def _server_cmd(self) -> list[str]:
        return [sys.executable, str(FIXTURE)]

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self) -> None:
        cmd = self._server_cmd()
        client = McpClient("mock", cmd[0], args=cmd[1:])
        try:
            await client.connect()
            names = [t["name"] for t in client.tools]
            assert names == ["echo", "fail"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_call_echo_tool(self) -> None:
        cmd = self._server_cmd()
        client = McpClient("mock", cmd[0], args=cmd[1:])
        try:
            await client.connect()
            text, is_error = await client.call_tool("echo", {"message": "hi"})
            assert text == "hi"
            assert is_error is False
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_call_fail_tool_returns_error(self) -> None:
        cmd = self._server_cmd()
        client = McpClient("mock", cmd[0], args=cmd[1:])
        try:
            await client.connect()
            text, is_error = await client.call_tool("fail", {"message": "boom"})
            assert text == "boom"
            assert is_error is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_command_not_found_raises_mcp_error(self) -> None:
        client = McpClient("bad", "/nonexistent/cmd-xyz", args=[])
        with pytest.raises(McpError):
            await client.connect()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        cmd = self._server_cmd()
        client = McpClient("mock", cmd[0], args=cmd[1:])
        await client.connect()
        await client.close()
        await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_are_dispatched_to_matching_requests(self) -> None:
        cmd = self._server_cmd()
        client = McpClient("mock", cmd[0], args=cmd[1:])
        try:
            await client.connect()
            messages = [f"message-{i}" for i in range(40)]
            results = await asyncio.wait_for(
                asyncio.gather(*(
                    client.call_tool("echo", {"message": message})
                    for message in messages
                )),
                timeout=5,
            )
            assert results == [(message, False) for message in messages]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_rpc_times_out_when_server_does_not_respond(self, tmp_path: Path) -> None:
        server = tmp_path / "silent_server.py"
        server.write_text(
            "import json, sys, time\n"
            "for line in sys.stdin:\n"
            " msg = json.loads(line)\n"
            " if msg.get('method') == 'initialize':\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{"
            "'protocolVersion':'2025-03-26','capabilities':{},'serverInfo':{"
            "'name':'silent','version':'1'}}}), flush=True)\n"
            " elif msg.get('method') == 'tools/list':\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'tools':[]}}), flush=True)\n"
            " elif msg.get('method') == 'tools/call':\n"
            "  time.sleep(1)\n"
            "  print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{"
            "'content':[]}}), flush=True)\n",
            encoding="utf-8",
        )
        client = McpClient(
            "slow", sys.executable, args=[str(server)], request_timeout_secs=0.5
        )
        try:
            await client.connect()
            client.request_timeout_secs = 0.05
            with pytest.raises(McpError, match="timed out"):
                await client.call_tool("slow", {})
        finally:
            await client.close()


class TestMcpClientManager:
    def _server_cmd(self) -> list[str]:
        return [sys.executable, str(FIXTURE)]

    @pytest.mark.asyncio
    async def test_connect_and_register(self) -> None:
        cmd = self._server_cmd()
        servers = {"mock": {"command": cmd[0], "args": cmd[1:]}}
        manager = McpClientManager()
        try:
            await manager.connect_all(servers)
            assert "mock" in manager.clients
            registry = ToolRegistry()
            count = manager.register_all(registry)
            assert count == 2
            assert "mcp__mock__echo" in registry.names()
            assert "mcp__mock__fail" in registry.names()
        finally:
            await manager.close_all()

    @pytest.mark.asyncio
    async def test_connect_all_skips_failed_server(self) -> None:
        servers = {"bad": {"command": "/nonexistent/cmd-xyz", "args": []}}
        manager = McpClientManager()
        await manager.connect_all(servers)
        assert "bad" not in manager.clients
        await manager.close_all()

    @pytest.mark.asyncio
    async def test_register_all_empty_when_no_clients(self) -> None:
        manager = McpClientManager()
        registry = ToolRegistry()
        assert manager.register_all(registry) == 0

    @pytest.mark.asyncio
    async def test_close_all_clears_clients(self) -> None:
        cmd = self._server_cmd()
        servers = {"mock": {"command": cmd[0], "args": cmd[1:]}}
        manager = McpClientManager()
        await manager.connect_all(servers)
        await manager.close_all()
        assert manager.clients == {}
