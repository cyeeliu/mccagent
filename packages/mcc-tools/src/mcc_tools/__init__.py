"""内建工具集合。

提供工具注册表与 read/write/edit/bash/grep/glob/list 等工具实现。
"""

from __future__ import annotations

from mcc_core import Tool


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


__all__ = ["ToolRegistry"]
__version__ = "0.1.0"
