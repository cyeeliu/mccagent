"""edit_file 工具：精确替换文件中的文本片段。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import resolve


class EditFileTool:
    """替换已有文件中的精确文本，通过原子重命名写入。"""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Replace exact text in an existing file. "
            "`old_string` must be unique (or set replace_all=true). "
            "Uses atomic rename to prevent partial writes."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path", "old_string", "new_string"],
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path_str = input.get("path")
        old_string = input.get("old_string")
        new_string = input.get("new_string")
        replace_all = input.get("replace_all", False)
        if not isinstance(path_str, str):
            return ToolOutput.err("missing or invalid required field: path")
        if not isinstance(old_string, str):
            return ToolOutput.err("missing or invalid required field: old_string")
        if not isinstance(new_string, str):
            return ToolOutput.err("missing or invalid required field: new_string")

        if old_string == new_string:
            return ToolOutput.err("old_string and new_string are identical — nothing to do")

        path = resolve(ctx.cwd, path_str)
        try:
            body = await _read_text(path)
        except OSError as e:
            return ToolOutput.err(f"read {path}: {e}")

        count = body.count(old_string)
        if count == 0:
            return ToolOutput.err(
                "old_string not found in file — check for whitespace or line-ending differences"
            )
        if count > 1 and not replace_all:
            return ToolOutput.err(
                f"old_string matches {count} times — add surrounding context to make it unique, "
                "or set replace_all=true to replace all occurrences"
            )

        new_body = body.replace(old_string, new_string) if replace_all else body.replace(
            old_string, new_string, 1
        )

        tmp = path.with_suffix(path.suffix + ".mcctmp")
        try:
            await _write_text(tmp, new_body)
            await _rename(tmp, path)
        except OSError as e:
            with contextlib.suppress(OSError):
                await _unlink(tmp)
            return ToolOutput.err(f"atomic write failed: {e}")

        suffix = "" if count == 1 else "s"
        return ToolOutput.ok(f"edited: {path} ({count} replacement{suffix})")


async def _read_text(path: Path) -> str:
    import asyncio

    return await asyncio.to_thread(path.read_text, "utf-8")


async def _write_text(path: Path, data: str) -> None:
    import asyncio

    await asyncio.to_thread(path.write_text, data, "utf-8")


async def _rename(src: Path, dst: Path) -> None:
    import asyncio

    await asyncio.to_thread(src.rename, dst)


async def _unlink(path: Path) -> None:
    import asyncio

    await asyncio.to_thread(path.unlink)
