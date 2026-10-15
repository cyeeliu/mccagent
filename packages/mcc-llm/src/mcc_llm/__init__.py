"""LLM provider 抽象与实现。

定义 LlmProvider 协议与流式事件类型，
支持 Anthropic（主）与 OpenAI 兼容端点（fallback）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mcc_core import Message, Usage


@dataclass
class StreamEvent:
    """LLM 流式输出的事件。"""

    kind: str
    text: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None


@dataclass
class LlmResponse:
    """非流式完整响应。"""

    content: list[Any]
    usage: Usage
    stop_reason: str | None = None


@runtime_checkable
class LlmProvider(Protocol):
    """LLM provider 协议：complete (非流式) + stream (SSE 流式)。"""

    @property
    def model(self) -> str: ...

    async def complete(
        self, messages: list[Message], system: str | None = None, max_tokens: int = 4096
    ) -> LlmResponse: ...

    async def stream(
        self, messages: list[Message], system: str | None = None, max_tokens: int = 4096
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = ["LlmProvider", "LlmResponse", "StreamEvent"]
__version__ = "0.1.0"
