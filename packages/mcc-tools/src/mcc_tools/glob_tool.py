"""glob 工具：按 glob 模式查找文件，遵守 .gitignore。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import is_ignored, load_gitignore, resolve


class GlobTool:
    """按 glob 模式查找文件，按修改时间排序（最新在前）。"""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            'Find files by glob pattern (e.g. "src/**/*.py"). '
            "Returns paths sorted by modification time (newest first). "
            "Respects .gitignore."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Root directory (default: cwd)"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        pattern = input.get("pattern")
        if not isinstance(pattern, str):
            return ToolOutput.err("missing or invalid required field: pattern")
        path_str = input.get("path", ".")

        root = resolve(ctx.cwd, path_str)
        if not root.is_dir():
            return ToolOutput.err(f"not a directory: {root}")

        spec = load_gitignore(root)
        return await asyncio.to_thread(self._find, root, pattern, spec)

    def _find(
        self, root: Path, pattern: str, spec: Any
    ) -> ToolOutput:
        entries: list[tuple[float, Path]] = []
        try:
            matched = list(root.glob(pattern))
        except OSError as e:
            return ToolOutput.err(f"glob: {e}")

        for p in matched:
            if not p.is_file():
                continue
            if is_ignored(p, root, spec):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            entries.append((mtime, p))

        entries.sort(key=lambda x: x[0], reverse=True)

        if not entries:
            return ToolOutput.ok(f"No files match `{pattern}`.")

        out_parts: list[str] = []
        for _, p in entries[:500]:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            out_parts.append(str(rel))
        if len(entries) > 500:
            out_parts.append(f"… [{len(entries) - 500} more files]")
        return ToolOutput.ok("\n".join(out_parts) + "\n")
