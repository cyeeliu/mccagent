"""Mock 适配器测试：事件序列、字段与失败分支。"""

import pytest
from runner.adapters.base import RunRequest
from runner.adapters.mock import MockAdapter
from runner.events import (
    LlmChunk,
    LlmRequest,
    RunFinished,
    RunStarted,
    ToolCall,
    ToolResult,
)


def _req(task: str = "fix the divide function") -> RunRequest:
    return RunRequest(
        run_id="run-1", case_id="case-1", task=task, model="mock-model-v1"
    )


async def collect(task: str) -> list:
    adapter = MockAdapter(pace_s=0.0)
    return [ev async for ev in adapter.run(_req(task))]


async def test_event_sequence_and_seq_monotonic() -> None:
    events = [ev async for ev in MockAdapter(pace_s=0.0).run(_req())]
    types = [ev.type for ev in events]

    assert types[0] == "run_started"
    assert types[-1] == "run_finished"
    assert types.count("tool_call") == 3
    assert types.count("tool_result") == 3
    assert types.count("llm_request") == 3
    assert types.count("llm_chunk") == 6

    seqs = [ev.seq for ev in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


async def test_run_started_fields() -> None:
    events = await collect("fix the divide function")
    started = events[0]
    assert isinstance(started, RunStarted)
    assert started.run_id == "run-1"
    assert started.case_id == "case-1"
    assert started.task == "fix the divide function"
    assert started.model == "mock-model-v1"
    assert started.schema_version == 1


async def test_three_turn_tool_plan() -> None:
    events = await collect("fix the divide function")
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert [c.tool_name for c in calls] == ["grep", "edit", "bash"]
    assert [c.turn for c in calls] == [0, 1, 2]

    results = [e for e in events if isinstance(e, ToolResult)]
    assert all(not r.is_error for r in results)
    assert results[-1].output == "test result: ok. 12 passed"

    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.status == "passed"
    assert finished.score == 1.0
    assert finished.turns == 3
    assert finished.cost_usd is not None and 0 < finished.cost_usd < 1


async def test_task_with_fail_simulates_failure() -> None:
    events = await collect("make the tests FAIL")
    results = [e for e in events if isinstance(e, ToolResult)]
    assert results[-1].is_error is True
    assert "FAILED" in results[-1].output

    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.status == "failed"
    assert finished.score == 0.0


async def test_llm_events_shape() -> None:
    events = await collect("fix the divide function")
    reqs = [e for e in events if isinstance(e, LlmRequest)]
    assert reqs[0].input_tokens == 1200
    assert reqs[1].input_tokens == 2000

    chunks = [e for e in events if isinstance(e, LlmChunk)]
    assert any("grep" in c.delta for c in chunks)


@pytest.mark.parametrize("task", ["fail", "please FAIL this", "Failure case"])
async def test_fail_detection_is_case_insensitive(task: str) -> None:
    events = [ev async for ev in MockAdapter(pace_s=0.0).run(_req(task))]
    assert events[-1].status == "failed"
