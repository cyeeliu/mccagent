"""事件契约测试：Runner 侧事件必须与 schemas/trace-event.schema.json 一致。"""

import json
import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from runner.events import (
    LlmChunk,
    LlmRequest,
    RunFinished,
    RunStarted,
    Seq,
    ToolCall,
    ToolResult,
    trace_event_adapter,
)

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "trace-event.schema.json"

MODEL_BY_TYPE = {
    "run_started": RunStarted,
    "llm_request": LlmRequest,
    "llm_chunk": LlmChunk,
    "tool_call": ToolCall,
    "tool_result": ToolResult,
    "run_finished": RunFinished,
}


@pytest.fixture(scope="module")
def schema_event_types() -> set[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}


@pytest.mark.parametrize("event_type", MODEL_BY_TYPE)
def test_event_types_match_schema(event_type: str, schema_event_types: set[str]) -> None:
    assert event_type in schema_event_types


def test_no_extra_models_beyond_schema(schema_event_types: set[str]) -> None:
    assert set(MODEL_BY_TYPE) == schema_event_types


def test_tool_call_round_trip() -> None:
    seq = Seq()
    ev = ToolCall(seq=seq.next(), turn=1, call_id="c1", tool_name="bash",
                  args={"command": "ls"})
    raw = json.loads(ev.model_dump_json())
    assert raw["type"] == "tool_call"
    assert raw["schema_version"] == 1
    back = trace_event_adapter.validate_python(raw)
    assert isinstance(back, ToolCall)
    assert back.args == {"command": "ls"}


def test_unknown_type_rejected() -> None:
    with pytest.raises(ValidationError):
        trace_event_adapter.validate_python(
            {"schema_version": 1, "seq": 0, "ts": 0, "type": "mystery"}
        )


def test_seq_monotonic() -> None:
    s = Seq()
    assert [s.next() for _ in range(3)] == [0, 1, 2]


def test_run_finished_status_validated() -> None:
    with pytest.raises(ValidationError):
        RunFinished.model_validate({"seq": 0, "status": "not-a-status"})


def test_schema_version_validated() -> None:
    with pytest.raises(ValidationError):
        RunFinished.model_validate({"seq": 0, "schema_version": 2, "status": "passed"})


def test_ts_defaults_to_now() -> None:
    before = time.time()
    ev = RunFinished.model_validate({"seq": 0, "status": "passed"})
    after = time.time()
    assert before <= ev.ts <= after


def test_run_started_serializes_optional_model() -> None:
    ev = RunStarted.model_validate(
        {"seq": 0, "run_id": "r1", "case_id": "c1", "task": "t"}
    )
    raw = json.loads(ev.model_dump_json())
    assert raw["model"] is None
    ev2 = RunStarted.model_validate(
        {"seq": 0, "run_id": "r1", "case_id": "c1", "task": "t", "model": "m1"}
    )
    assert ev2.model == "m1"


def test_jsonl_line_is_single_line() -> None:
    ev = LlmChunk(seq=0, turn=0, delta="multi\nline\ntext")
    line = ev.model_dump_json()
    assert "\n" not in line
