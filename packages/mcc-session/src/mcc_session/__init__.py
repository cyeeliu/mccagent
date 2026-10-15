"""会话持久化与记忆。

定义会话元数据与轮次快照类型，支持 JSONL 落盘与回放。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mcc_core import Usage
from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    id: str
    started_at: datetime
    cwd: str
    turns: int = 0
    cost_usd: float = 0.0
    status: str = "active"
    title: str | None = None


class TurnSnapshot(BaseModel):
    ts: datetime
    iteration: int
    request_messages: list[dict[str, Any]] = Field(default_factory=list)
    assistant_blocks: list[dict[str, Any]] = Field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    cost_usd: float = 0.0


__all__ = ["SessionMeta", "TurnSnapshot"]
__version__ = "0.1.0"
