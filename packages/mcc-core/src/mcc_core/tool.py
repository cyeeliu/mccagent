"""Tool 协议与执行上下文。

用 typing.Protocol 定义同步可检查的鸭子类型接口，
execute 为 async，由 harness 并行调度。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolContext:
    cwd: Path
    session_id: str
    depth: int = 0


@dataclass
class ToolOutput:
    content: str
    is_error: bool = False

    @classmethod
    def ok(cls, content: str) -> ToolOutput:
        return cls(content=content, is_error=False)

    @classmethod
    def err(cls, content: str) -> ToolOutput:
        return cls(content=content, is_error=True)


@runtime_checkable
class Tool(Protocol):
    """工具协议：每个内建工具实现此接口。"""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def input_schema(self) -> dict[str, Any]: ...

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput: ...


@dataclass
class ToolCall:
    """一次工具调用的请求（从 LLM 的 tool_use block 解析而来）。"""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
