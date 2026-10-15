"""消息与内容块类型。

与 Anthropic Messages API 的 JSON 形状对齐：
- TextBlock   → {"type": "text", "text": "..."}
- ToolUseBlock → {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
- ToolResultBlock → {"type": "tool_result", "tool_use_id": "...", "content": "..."}
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class CacheControl(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    kind: str = Field(alias="type")


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: CacheControl | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class Message(BaseModel):
    role: Role
    content: list[ContentBlock] = Field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.user, content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.assistant, content=[TextBlock(text=text)])

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role=Role.system, content=[TextBlock(text=text)])


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
        )
