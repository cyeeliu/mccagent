"""轨迹事件：一次评测运行的完整过程记录（schema_version = 1）。

与 ``schemas/trace-event.schema.json`` 对齐，由契约测试钉住一致性；
字段变更必须同步 schema 与测试并提升版本号。

六类事件覆盖一轮 Agent 执行的全部可观测过程：
run_started → (llm_request → llm_chunk → tool_call → tool_result)* → run_finished
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
    # error / timeout 的具体原因（如 LLM 认证失败、网关不可达）。
    # 没有它，失败 run 只剩光秃秃的 status，无从排障
    reason: str | None = None


TraceEvent = Annotated[
    RunStarted | LlmRequest | LlmChunk | ToolCall | ToolResult | RunFinished,
    Field(discriminator="type"),
]

trace_event_adapter: TypeAdapter[TraceEvent] = TypeAdapter(TraceEvent)


class Seq:
    """run 内单调序号发生器；事件排序与合并的锚点。"""

    def __init__(self) -> None:
        self._n = -1

    def next(self) -> int:
        self._n += 1
        return self._n
