"""LLM provider 抽象与实现。

定义 LlmProvider 协议与流式事件类型，
支持 Anthropic（主）与 OpenAI 兼容端点（fallback）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mcc_core import ContentBlock, Message, Usage

# ---------- 流式事件 ----------


@dataclass
class TextDelta:
    """文本增量片段。"""

    text: str


@dataclass
class ToolUseStart:
    """工具调用开始。"""

    id: str
    name: str


@dataclass
class ToolUseInputDelta:
    """工具输入 JSON 增量片段。"""

    partial_json: str


@dataclass
class MessageStop:
    """消息结束，携带停止原因与 token 用量。"""

    stop_reason: str
    usage: Usage = field(default_factory=Usage)


StreamEvent = TextDelta | ToolUseStart | ToolUseInputDelta | MessageStop


# ---------- 请求 / 响应 ----------


@dataclass
class CompleteRequest:
    """LLM 完成请求。"""

    model: str
    max_tokens: int
    messages: list[Message]
    system: str | None = None
    temperature: float | None = None
    tools: list[dict[str, Any]] | None = None


@dataclass
class MessageResponse:
    """非流式完整响应。"""

    content: list[ContentBlock]
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


# ---------- Provider 协议 ----------


@runtime_checkable
class LlmProvider(Protocol):
    """LLM provider 协议：complete (非流式) + stream (SSE 流式)。"""

    async def complete(self, req: CompleteRequest) -> MessageResponse: ...

    def stream(self, req: CompleteRequest) -> AsyncIterator[StreamEvent]: ...


# ---------- 工厂 ----------


def auto_provider_from_env() -> LlmProvider:
    """根据环境变量自动挑选 provider。

    优先 ANTHROPIC_API_KEY，其次 OPENAI_API_KEY。
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        from mcc_llm.anthropic import AnthropicClient

        return AnthropicClient.from_env()
    if os.environ.get("OPENAI_API_KEY"):
        from mcc_llm.openai import OpenAiClient

        return OpenAiClient.from_env()
    msg = (
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
        "(and optionally OPENAI_BASE_URL for a compatible gateway)."
    )
    raise RuntimeError(msg)


__all__ = [
    "CompleteRequest",
    "LlmProvider",
    "MessageResponse",
    "MessageStop",
    "StreamEvent",
    "TextDelta",
    "ToolUseInputDelta",
    "ToolUseStart",
    "auto_provider_from_env",
]
__version__ = "0.1.0"
