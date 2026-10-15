"""内建工具集合。

提供工具注册表与 read/write/edit/bash/grep/glob/list 等工具实现。
"""

from __future__ import annotations

from typing import Any

from mcc_core import Tool

from .edit_file import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .list_dir import ListDirTool
from .read_file import ReadFileTool
from .run_bash import RunBashTool
from .write_file import WriteFileTool


class ToolRegistry:
    """工具注册表：name → Tool 实例。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def as_api_schema(self) -> list[dict[str, Any]]:
        """返回所有工具的 API schema，按 name 排序。"""
        tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema(),
            }
            for t in self._tools.values()
        ]
        tools.sort(key=lambda t: t["name"])
        return tools

    def subset(self, allowed: list[str]) -> ToolRegistry:
        """返回仅包含 allowed 列表中工具的新注册表。"""
        r = ToolRegistry()
        for n in allowed:
            t = self._tools.get(n)
            if t is not None:
                r._tools[n] = t
        return r


def default_registry() -> ToolRegistry:
    """创建包含全部内建工具的注册表。"""
    r = ToolRegistry()
    r.register(ReadFileTool())
    r.register(ListDirTool())
    r.register(WriteFileTool())
    r.register(EditFileTool())
    r.register(RunBashTool())
    r.register(GrepTool())
    r.register(GlobTool())
    return r


__all__ = [
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "ListDirTool",
    "ReadFileTool",
    "RunBashTool",
    "ToolRegistry",
    "WriteFileTool",
    "default_registry",
]
__version__ = "0.1.0"
