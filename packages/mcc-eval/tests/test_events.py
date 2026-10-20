"""轨迹事件测试：schema 契约钉住、序列化 roundtrip、序号发生器。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcc_eval.events import (
    LlmChunk,
    RunFinished,
    RunStarted,
    Seq,
    ToolCall,
    ToolResult,
    trace_event_adapter,
)

SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "trace-event.schema.json"


def _schema_types() -> set[str]:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        schema = json.load(fh)
    return {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}


def test_schema_covers_all_event_types() -> None:
    """schema oneOf 与 pydantic 判别式联合必须一一对应。"""
    expected = {"run_started", "llm_request", "llm_chunk", "tool_call", "tool_result",
                "run_finished"}
    assert _schema_types() == expected


@pytest.mark.parametrize(
    "event",
    [
        RunStarted(seq=0, run_id="r", case_id="c", task="t", model="m"),
        LlmChunk(seq=1, turn=0, delta="hi"),
        ToolCall(seq=2, turn=0, call_id="x", tool_name="bash", args="pytest"),
        ToolResult(seq=3, call_id="x", output="ok"),
        RunFinished(seq=4, status="passed", score=1.0),
    ],
    ids=lambda e: type(e).__name__,
)
def test_event_json_roundtrip(event) -> None:
    """dump→validate 往返保持类型与字段（trace JSONL 的读写契约）。"""
    raw = json.loads(event.model_dump_json())
    assert trace_event_adapter.validate_python(raw) == event


def test_seq_is_strictly_monotonic() -> None:
    seq = Seq()
    values = [seq.next() for _ in range(5)]
    assert values == [0, 1, 2, 3, 4]


def test_run_finished_status_vocabulary() -> None:
    """终态词汇表：passed / failed / error / timeout。"""
    for status in ("passed", "failed", "error", "timeout"):
        assert RunFinished(seq=0, status=status).status == status
    with pytest.raises(ValueError):
        RunFinished(seq=0, status="running")
