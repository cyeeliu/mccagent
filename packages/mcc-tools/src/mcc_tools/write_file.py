"""write_file 工具：创建新文件，拒绝覆盖已存在文件。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import resolve


class WriteFileTool:
    """创建新文件。如果文件已存在则失败，需用 edit_file 修改。"""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create a new file. Fails if the file already exists. "
            "Use edit_file to modify an existing file."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path_str = input.get("path")
        content = input.get("content")
        if not isinstance(path_str, str):
            return ToolOutput.err("missing or invalid required field: path")
        if not isinstance(content, str):
            return ToolOutput.err("missing or invalid required field: content")

        path = resolve(ctx.cwd, path_str)
        if path.exists():
            return ToolOutput.err(
                f"refusing to overwrite {path}; use edit_file to modify existing files"
            )

        parent = path.parent
        if parent and not parent.exists():
            try:
                await _mkdir(parent)
            except OSError as e:
                return ToolOutput.err(f"mkdir {parent}: {e}")

        try:
            await _write_bytes(path, content.encode("utf-8"))
        except OSError as e:
            return ToolOutput.err(f"write {path}: {e}")

        return ToolOutput.ok(f"written: {path} ({len(content)} bytes)")


async def _mkdir(path: Path) -> None:
    import asyncio

    await asyncio.to_thread(lambda: path.mkdir(parents=True))


async def _write_bytes(path: Path, data: bytes) -> None:
    import asyncio

    await asyncio.to_thread(path.write_bytes, data)
