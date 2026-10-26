"""Mock MCP server for testing.

Reads newline-delimited JSON-RPC 2.0 from stdin, writes responses to stdout.
Implements initialize / notifications/initialized / tools/list / tools/call.

Exposes two tools:
- echo: returns {"text": <message>} from input {"message": str}
- fail: returns isError=True with the given message
"""

from __future__ import annotations

import json
import sys


def handle_initialize(_params: dict) -> dict:
    return {
        "protocolVersion": "2025-03-26",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "mock-mcp", "version": "0.1.0"},
    }


def handle_tools_list(_params: dict) -> dict:
    return {
        "tools": [
            {
                "name": "echo",
                "description": "Echo back the message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "fail",
                "description": "Return an error result",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                },
            },
        ]
    }


def handle_tools_call(params: dict) -> dict:
    name = params.get("name", "")
    args = params.get("arguments", {})
    if name == "echo":
        return {
            "content": [{"type": "text", "text": args.get("message", "")}],
            "isError": False,
        }
    if name == "fail":
        return {
            "content": [{"type": "text", "text": args.get("message", "error")}],
            "isError": True,
        }
    return {
        "content": [{"type": "text", "text": f"unknown tool: {name}"}],
        "isError": True,
    }


HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method", "")
        req_id = msg.get("id")
        if method in HANDLERS:
            try:
                result = HANDLERS[method](msg.get("params", {}))
                if req_id is not None:
                    sys.stdout.write(
                        json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
                        + "\n"
                    )
                    sys.stdout.flush()
            except Exception as e:
                if req_id is not None:
                    sys.stdout.write(
                        json.dumps({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32603, "message": str(e)},
                        })
                        + "\n"
                    )
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
