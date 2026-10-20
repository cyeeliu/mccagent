"""运行轨迹事件的 Runner 侧定义。

与 ``schemas/trace-event.schema.json`` 对齐（schema_version = 1），
由契约测试钉住与 schema 的一致性；字段变更必须同步三端并提升版本号。
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

SCHEMA_VERSION: Literal[1] = 1


class _Base(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    seq: int
    ts: float = Field(default_factory=time.time)


class RunStarted(_Base):
    type: Literal["run_started"] = "run_started"
    run_id: str
    case_id: str
    task: str
    model: str | None = None


class LlmRequest(_Base):
    type: Literal["llm_request"] = "llm_request"
    turn: int
    input_tokens: int = 0


class LlmChunk(_Base):
    type: Literal["llm_chunk"] = "llm_chunk"
    turn: int
    delta: str


class ToolCall(_Base):
    type: Literal["tool_call"] = "tool_call"
    turn: int
    call_id: str
    tool_name: str
    args: object = None


class ToolResult(_Base):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str
    is_error: bool = False
    duration_ms: int = 0


class RunFinished(_Base):
    type: Literal["run_finished"] = "run_finished"
    status: Literal["passed", "failed", "error", "timeout"]
    score: float | None = None
    cost_usd: float | None = None
    turns: int = 0


TraceEvent = Annotated[
    RunStarted | LlmRequest | LlmChunk | ToolCall | ToolResult | RunFinished,
    Field(discriminator="type"),
]

trace_event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class Seq:
    """run 内单调序号发生器；前端实时/历史轨迹合并的锚点。"""

    def __init__(self) -> None:
        self._n = -1

    def next(self) -> int:
        self._n += 1
        return self._n
