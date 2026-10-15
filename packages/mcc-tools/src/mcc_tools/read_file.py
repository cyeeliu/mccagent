"""read_file 工具：读取 UTF-8 文本文件并添加行号。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import resolve


class ReadFileTool:
    """读取 UTF-8 文本文件，输出带行号的内容。"""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a UTF-8 text file with line numbers. "
            "Use `offset` and `limit` to page through large files."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "Relative or absolute path"},
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "First line (0-based)",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2000,
                    "description": "Max lines to return",
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path_str = input.get("path")
        if not isinstance(path_str, str):
            return ToolOutput.err("missing or invalid required field: path")
        offset = input.get("offset", 0)
        limit = input.get("limit", 2000)

        path = resolve(ctx.cwd, path_str)
        try:
            body = await _read_text(path)
        except OSError as e:
            return ToolOutput.err(f"read {path}: {e}")

        lines = body.splitlines()
        total = len(lines)
        start = min(offset, total)
        end = min(start + limit, total)

        out_parts: list[str] = []
        for i, line in enumerate(lines[start:end]):
            out_parts.append(f"{start + i + 1:>6}\t{line}")
        out = "\n".join(out_parts)
        if out:
            out += "\n"
        if end < total:
            out += f"\n… [{total - end} more lines, use offset={end}]\n"
        return ToolOutput.ok(out)


async def _read_text(path: Path) -> str:
    import asyncio

    return await asyncio.to_thread(path.read_text, "utf-8")
