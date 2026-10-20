"""领域类型契约测试：事件模型必须与 schemas/trace-event.schema.json 一致。"""

import json
from pathlib import Path

import pytest
from eval_server.domain import (
    EVENT_TYPES,
    LlmChunk,
    LlmRequest,
    RunFinished,
    RunStarted,
    ToolCall,
    ToolResult,
    is_milestone,
    trace_event_adapter,
    validate_for_run,
)
from pydantic import ValidationError

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "trace-event.schema.json"

MINIMAL_PAYLOADS: dict[str, dict] = {
    "run_started": {"run_id": "r1", "case_id": "c1", "task": "do it"},
    "llm_request": {"turn": 0},
    "llm_chunk": {"turn": 0, "delta": "hello"},
    "tool_call": {"turn": 0, "call_id": "c1", "tool_name": "bash", "args": {"command": "ls"}},
    "tool_result": {"call_id": "c1", "output": "done", "is_error": False, "duration_ms": 12},
    "run_finished": {"status": "passed", "score": 0.9, "cost_usd": 0.01, "turns": 3},
}

EVENT_MODEL_BY_TYPE = {
    "run_started": RunStarted,
    "llm_request": LlmRequest,
    "llm_chunk": LlmChunk,
    "tool_call": ToolCall,
    "tool_result": ToolResult,
    "run_finished": RunFinished,
}


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema_event_types(schema: dict) -> set[str]:
    return {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_event_types_cover_schema_branches(event_type: str, schema_event_types: set[str]) -> None:
    assert event_type in schema_event_types


def test_no_extra_model_beyond_schema(schema_event_types: set[str]) -> None:
    assert set(EVENT_MODEL_BY_TYPE) == schema_event_types


@pytest.mark.parametrize("event_type", EVENT_TYPES)
def test_round_trip_per_type(event_type: str) -> None:
    ev = trace_event_adapter.validate_python({"seq": 0, "type": event_type,
                                              **MINIMAL_PAYLOADS[event_type]})
    dumped = json.loads(ev.model_dump_json())
    assert dumped["schema_version"] == 1
    assert dumped["type"] == event_type
    assert isinstance(dumped["ts"], float)
    back = trace_event_adapter.validate_python(dumped)
    assert type(back) is type(ev)


def test_unknown_event_type_rejected() -> None:
    with pytest.raises(ValidationError):
        trace_event_adapter.validate_python({"seq": 0, "ts": 0.0, "type": "mystery"})


@pytest.mark.parametrize("bad_status", ["ok", "crashed", "PASSED", ""])
def test_run_finished_status_validated(bad_status: str) -> None:
    with pytest.raises(ValidationError):
        RunFinished.model_validate({"seq": 0, "status": bad_status})


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", 2), ("schema_version", 0)],
)
def test_schema_version_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        RunFinished.model_validate({"seq": 0, field: value, "status": "passed"})


def test_seq_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        RunFinished.model_validate({"seq": -1, "status": "passed"})


def test_validate_for_run_rejects_foreign_run_id() -> None:
    ev = trace_event_adapter.validate_python(
        {"seq": 0, "type": "run_started", "run_id": "other", "case_id": "c", "task": "t"}
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_for_run(ev, "expected")


def test_validate_for_run_accepts_matching_run_id() -> None:
    ev = trace_event_adapter.validate_python(
        {"seq": 0, "type": "run_started", "run_id": "r1", "case_id": "c", "task": "t"}
    )
    validate_for_run(ev, "r1")


def test_validate_for_run_ignores_non_start_events() -> None:
    ev = trace_event_adapter.validate_python(
        {"seq": 1, "type": "llm_chunk", "turn": 0, "delta": "x"}
    )
    validate_for_run(ev, "any-run")


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("run_started", True),
        ("run_finished", True),
        ("llm_request", False),
        ("llm_chunk", False),
        ("tool_call", False),
        ("tool_result", False),
    ],
)
def test_is_milestone(event_type: str, expected: bool) -> None:
    ev = trace_event_adapter.validate_python(
        {"seq": 0, "type": event_type, **MINIMAL_PAYLOADS[event_type]}
    )
    assert is_milestone(ev) is expected
