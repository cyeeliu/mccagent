"""list_dir 工具：列出目录内容，遵守 .gitignore。"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import is_ignored, load_gitignore, resolve


class ListDirTool:
    """列出目录内容，遵守 .gitignore，返回树状列表。"""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "List directory contents. Respects .gitignore. "
            "Returns a tree-like listing up to max_depth levels."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"},
                "max_depth": {
                    "type": "integer",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path_str = input.get("path")
        if not isinstance(path_str, str):
            return ToolOutput.err("missing or invalid required field: path")
        max_depth = input.get("max_depth", 3)

        root = resolve(ctx.cwd, path_str)
        if not root.is_dir():
            return ToolOutput.err(f"not a directory: {root}")

        spec = load_gitignore(root)
        out_parts: list[str] = []
        count = 0
        queue: deque[tuple[Path, int]] = deque([(root, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth > 0:
                if is_ignored(current, root, spec):
                    continue
                indent = "  " * (depth - 1)
                out_parts.append(f"{indent}{current.name}/")
                count += 1
                if count >= 2000:
                    out_parts.append("… [truncated at 2000 entries]")
                    return ToolOutput.ok("\n".join(out_parts) + "\n")
            if depth >= max_depth:
                continue
            try:
                children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except OSError as e:
                return ToolOutput.err(f"list {current}: {e}")
            for child in children:
                if child.is_dir():
                    queue.append((child, depth + 1))
                else:
                    if is_ignored(child, root, spec):
                        continue
                    indent = "  " * depth
                    out_parts.append(f"{indent}{child.name}")
                    count += 1
                    if count >= 2000:
                        out_parts.append("… [truncated at 2000 entries]")
                        return ToolOutput.ok("\n".join(out_parts) + "\n")

        return ToolOutput.ok("\n".join(out_parts) + "\n" if out_parts else "")
