"""会话元数据与轮次快照类型。"""

from __future__ import annotations

from datetime import datetime

from mcc_core import ContentBlock, Message, Usage
from pydantic import BaseModel, Field


class SessionMeta(BaseModel):
    """会话元数据：随每轮 record 重写，供列表与恢复使用。"""

    id: str
    started_at: datetime
    cwd: str
    turns: int = 0
    cost_usd: float = 0.0
    status: str = "running"
    title: str | None = None


class TurnSnapshot(BaseModel):
    """单轮快照：JSONL 中每行一个。

    cost_usd 由 AgentLoop 按其定价表计算后写入；
    字段缺失时按 0 处理，保证旧流水可继续解析。
    """

    ts: datetime
    iteration: int
    request_messages: list[Message] = Field(default_factory=list)
    assistant_blocks: list[ContentBlock] = Field(default_factory=list)
    tool_outputs: list[tuple[str, str, bool]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    cost_usd: float = 0.0


__all__ = ["SessionMeta", "TurnSnapshot"]
