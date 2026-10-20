"""执行器测试：run_case 全链路、超时 / 异常路径、并发批次与汇总统计。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from mcc_eval.adapters import MockAdapter, RunRequest
from mcc_eval.datasets import Case
from mcc_eval.events import RunFinished, TraceEvent
from mcc_eval.runner import (
    BatchResult,
    CaseResult,
    new_batch_dir,
    run_batch,
    run_case,
    snapshot_from_events,
)


class _WriterAdapter:
    """发一轮「写文件 + 汇报」轨迹并真实落盘 solution.py。"""

    def __init__(self, code: str) -> None:
        self.code = code

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        from mcc_eval.events import LlmChunk, ToolCall, ToolResult

        work_dir = Path(req.work_dir or ".")
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "solution.py").write_text(self.code, encoding="utf-8")
        yield LlmChunk(seq=0, turn=0, delta="writing solution")
        yield ToolCall(seq=1, turn=0, call_id="w", tool_name="write_file",
                       args={"path": "solution.py"})
        yield ToolResult(seq=2, call_id="w", output="written", is_error=False)
        yield LlmChunk(seq=3, turn=0, delta="PASSED")
        yield RunFinished(seq=4, status="passed", score=None, cost_usd=0.01, turns=1)


class _ExplodingAdapter:
    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        from mcc_eval.events import RunStarted

        yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                         task=req.task, model=req.model)
        raise RuntimeError("adapter exploded")


class _SlowAdapter:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        from mcc_eval.events import RunStarted

        yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                         task=req.task, model=req.model)
        await asyncio.sleep(self.delay_s)
        yield RunFinished(seq=1, status="passed")


def _case(**overrides) -> Case:
    payload = {"case_id": "case-1", "task": "do work",
               "expectations": [{"type": "no_tool_errors", "weight": 1.0}]}
    payload.update(overrides)
    return Case(**payload)


# ─── run_case ────────────────────────────────────────────────────────────


async def test_run_case_assertion_scoring(tmp_path: Path) -> None:
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(), MockAdapter(), model="mock", batch_dir=batch_dir,
                            timeout_s=5)
    assert result.status == "passed"
    assert result.score == 1.0
    assert result.turns == 3
    assert result.cost_usd is not None

    trace = batch_dir / "runs" / "case-1.jsonl"
    lines = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines() if ln]
    assert lines[0]["type"] == "run_started"
    assert lines[-1]["type"] == "run_finished"
    assert (batch_dir / "work" / "case-1").is_dir()


async def test_run_case_execution_scoring(tmp_path: Path) -> None:
    """官方测试 + Agent 产物：写对 solution 判 passed，写错判 failed。"""
    check = "from solution import double as candidate\nassert candidate(3) == 6\n"
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)

    good = await run_case(
        _case(check=check), _WriterAdapter("def double(x):\n    return x * 2\n"),
        model="mock", batch_dir=batch_dir, timeout_s=10,
    )
    assert good.status == "passed"
    assert good.score == 1.0

    bad = await run_case(
        _case(case_id="case-2", check=check), _WriterAdapter("def double(x):\n    return x\n"),
        model="mock", batch_dir=batch_dir, timeout_s=10,
    )
    assert bad.status == "failed"
    assert bad.score == 0.0
    assert "execution check failed" in bad.reason


async def test_run_case_timeout_appends_terminal_event(tmp_path: Path) -> None:
    """超时补 RunFinished 落盘：轨迹末事件永远标记终态，且附 reason。"""
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(), _SlowAdapter(delay_s=5), model="mock",
                            batch_dir=batch_dir, timeout_s=0.05)

    assert result.status == "timeout"
    assert "timeout" in result.reason
    lines = [json.loads(ln) for ln in
             (batch_dir / "runs" / "case-1.jsonl").read_text(encoding="utf-8").splitlines()
             if ln]
    assert lines[-1]["type"] == "run_finished"
    assert lines[-1]["status"] == "timeout"


async def test_run_case_adapter_crash_reports_error(tmp_path: Path) -> None:
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(), _ExplodingAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5)
    assert result.status == "error"
    assert "adapter exploded" in result.reason
    assert result.score is None


# ─── run_batch 与统计 ────────────────────────────────────────────────────


async def test_run_batch_concurrency_and_summary(tmp_path: Path) -> None:
    cases = [_case(case_id=f"c{i}", task=f"task {i}" if i else "this will fail")
             for i in range(4)]
    batch_dir = new_batch_dir(tmp_path, "unit", 4, 4)
    batch = await run_batch(cases, MockAdapter(), dataset="unit", model="mock",
                            batch_dir=batch_dir, concurrency=2, timeout_s=5)

    assert batch.total == 4
    assert batch.passed == 3
    assert batch.failed == 1
    assert batch.errors == 0
    assert batch.pass_rate == 0.75
    assert batch.cost_usd > 0
    assert {r.case_id for r in batch.results} == {c.case_id for c in cases}

    data = batch.to_dict()
    assert data["summary"]["total"] == 4
    assert len(data["results"]) == 4


async def test_run_batch_progress_lines(tmp_path: Path, capsys) -> None:
    cases = [_case(case_id="only")]
    batch_dir = new_batch_dir(tmp_path, "unit", 1, 1)
    await run_batch(cases, MockAdapter(), dataset="unit", model="mock",
                    batch_dir=batch_dir, concurrency=1, timeout_s=5)
    err = capsys.readouterr().err
    assert "[1/1] only:" in err


# ─── snapshot 折叠 ──────────────────────────────────────────────────────


def test_snapshot_from_events_folds_pipeline() -> None:
    from mcc_eval.events import LlmChunk, ToolCall, ToolResult

    events = [
        ToolCall(seq=0, turn=0, call_id="a", tool_name="bash", args="pytest"),
        ToolResult(seq=1, call_id="a", output="boom", is_error=True),
        LlmChunk(seq=2, turn=0, delta="hello "),
        LlmChunk(seq=3, turn=0, delta="world"),
    ]
    snap = snapshot_from_events(events)
    assert len(snap.tool_calls) == 1
    assert snap.had_error
    assert "".join(snap.text_outputs) == "hello world"


def test_case_result_and_batch_result_dicts() -> None:
    r = CaseResult(case_id="x", status="failed", score=0.25, reason="why",
                   cost_usd=0.01, turns=2, duration_s=1.5, details=["✗ a"])
    d = r.to_dict()
    assert d["duration_s"] == 1.5 and d["details"] == ["✗ a"]

    b = BatchResult(batch_id="b", dataset="d", model="m", started_at="s",
                    finished_at="f", results=[r])
    assert b.to_dict()["summary"]["pass_rate"] == 0.0
