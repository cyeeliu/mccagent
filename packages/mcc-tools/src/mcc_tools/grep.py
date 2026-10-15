"""grep 工具：跨目录树的正则搜索，遵守 .gitignore。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import is_ignored, load_gitignore, resolve


class GrepTool:
    """在目录树中搜索正则模式，遵守 .gitignore，返回匹配行。"""

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search for a regex pattern across files in a directory. "
            "Respects .gitignore. Returns matching lines with file:line context."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current dir)",
                },
                "glob": {
                    "type": "string",
                    "description": 'File glob filter, e.g. "*.rs" or "**/*.toml"',
                },
                "case_insensitive": {"type": "boolean", "default": False},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 200,
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        pattern = input.get("pattern")
        if not isinstance(pattern, str):
            return ToolOutput.err("missing or invalid required field: pattern")
        path_str = input.get("path", ".")
        glob_filter = input.get("glob")
        case_insensitive = input.get("case_insensitive", False)
        max_results = input.get("max_results", 200)

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolOutput.err(f"invalid regex: {e}")

        search_root = resolve(ctx.cwd, path_str)
        if not search_root.exists():
            return ToolOutput.err(f"path not found: {search_root}")

        spec = load_gitignore(search_root if search_root.is_dir() else search_root.parent)

        return await asyncio.to_thread(
            self._search,
            regex,
            search_root,
            glob_filter,
            max_results,
            spec,
            pattern,
        )

    def _search(
        self,
        regex: re.Pattern[str],
        search_root: Path,
        glob_filter: str | None,
        max_results: int,
        spec: Any,
        pattern_str: str,
    ) -> ToolOutput:
        import fnmatch

        files: list[Path] = []
        if search_root.is_file():
            files = [search_root]
        else:
            for p in search_root.rglob("*"):
                if p.is_file():
                    if is_ignored(p, search_root, spec):
                        continue
                    if glob_filter and not fnmatch.fnmatch(p.name, glob_filter):
                        continue
                    files.append(p)

        out_parts: list[str] = []
        total = 0
        files_searched = 0
        truncated = False

        for fpath in files:
            files_searched += 1
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    out_parts.append(f"{fpath}:{lineno}: {line}")
                    total += 1
                    if total >= max_results:
                        truncated = True
                        break
            if truncated:
                break

        if total == 0:
            return ToolOutput.ok(
                f"No matches for `{pattern_str}` in {files_searched} file(s) searched."
            )
        if truncated:
            out_parts.append(
                f"\n… [results truncated at {max_results} matches — refine pattern or use glob]"
            )
        else:
            out_parts.append(f"\n{total} match(es) across {files_searched} file(s).")
        return ToolOutput.ok("\n".join(out_parts) + "\n")
