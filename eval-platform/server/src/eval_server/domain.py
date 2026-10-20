"""领域类型：运行轨迹事件模型。

与 ``schemas/trace-event.schema.json`` 对齐（schema_version = 1）。
该 schema 是 server / web / runner 三端的单一事实来源，
任何字段变更都必须同步修改 schema 并提升版本号，再由各端契约测试钉住。
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

TRACE_SCHEMA_VERSION: Literal[1] = 1

RunOutcome = Literal["passed", "failed", "error", "timeout"]
RUN_OUTCOMES: tuple[RunOutcome, ...] = ("passed", "failed", "error", "timeout")

EVENT_TYPES: tuple[str, ...] = (
    "run_started",
    "llm_request",
    "llm_chunk",
    "tool_call",
    "tool_result",
    "run_finished",
)


class _EventBase(BaseModel):
    schema_version: Literal[1] = TRACE_SCHEMA_VERSION
    seq: int = Field(ge=0, description="run 内单调递增序号，实时/历史合并的锚点")
    ts: float = Field(default_factory=time.time, description="unix 秒（含小数）")


class RunStarted(_EventBase):
    type: Literal["run_started"] = "run_started"
    run_id: str
    case_id: str
    task: str
    model: str | None = None


class LlmRequest(_EventBase):
    type: Literal["llm_request"] = "llm_request"
    turn: int = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)


class LlmChunk(_EventBase):
    type: Literal["llm_chunk"] = "llm_chunk"
    turn: int = Field(ge=0)
    delta: str


class ToolCall(_EventBase):
    type: Literal["tool_call"] = "tool_call"
    turn: int = Field(ge=0)
    call_id: str
    tool_name: str
    args: object = None


class ToolResult(_EventBase):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str
    is_error: bool = False
    duration_ms: int = Field(default=0, ge=0)


class RunFinished(_EventBase):
    type: Literal["run_finished"] = "run_finished"
    status: RunOutcome
    score: float | None = None
    cost_usd: float | None = None
    turns: int = Field(default=0, ge=0)


TraceEvent = Annotated[
    RunStarted | LlmRequest | LlmChunk | ToolCall | ToolResult | RunFinished,
    Field(discriminator="type"),
]

trace_event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


def validate_for_run(event: TraceEvent, run_id: str) -> None:
    """校验事件归属；run_started 携带的 run_id 与请求不符时抛出 ValueError。"""
    if isinstance(event, RunStarted) and event.run_id != run_id:
        raise ValueError(
            f"event run_id {event.run_id!r} does not match request run_id {run_id!r}"
        )


def is_milestone(event: TraceEvent) -> bool:
    """批次列表只关心起止事件；高频过程事件在此降采样。"""
    return isinstance(event, (RunStarted, RunFinished))
