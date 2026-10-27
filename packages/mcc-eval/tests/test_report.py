"""报告测试：产物落盘、markdown 呈现与 rebuild 重评分。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcc_eval.adapters import MockAdapter
from mcc_eval.datasets import Case
from mcc_eval.events import RunFinished
from mcc_eval.report import (
    load_cases,
    load_trace_events,
    rebuild,
    save_cases,
    write_report_md,
    write_results_json,
)
from mcc_eval.runner import BatchResult, CaseResult, new_batch_dir, run_case


def _batch(tmp_path: Path) -> BatchResult:
    return BatchResult(
        batch_id="b1",
        dataset="humaneval",
        model="test-model",
        started_at="2026-10-20T10:00:00+00:00",
        finished_at="2026-10-20T10:01:00+00:00",
        results=[
            CaseResult(case_id="ok", status="passed", score=1.0, reason="all good",
                       cost_usd=0.01, turns=2, duration_s=3.2),
            CaseResult(case_id="bad", status="failed", score=0.0,
                       reason="execution check failed", turns=1, duration_s=1.0,
                       details=["✗ tool_called: tool 'write_file' not called"]),
        ],
    )


def test_results_json_and_report_md(tmp_path: Path) -> None:
    batch = _batch(tmp_path)
    results_path = write_results_json(batch, tmp_path)
    report_path = write_report_md(batch, tmp_path, title="HumanEval")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["summary"] == {
        "total": 2, "passed": 1, "failed": 1, "errors": 0,
        "pass_rate": 0.5, "cost_usd": 0.01,
        "cases": 2, "stable_cases": 1, "flaky_cases": 0,
    }

    text = report_path.read_text(encoding="utf-8")
    assert "# 评测报告：b1" in text
    assert "HumanEval" in text
    assert "| ok | passed | 1.00 |" in text
    assert "## 失败理由" in text
    assert "✗ tool_called" in text


def test_report_md_without_failures(tmp_path: Path) -> None:
    batch = BatchResult("b2", "mbpp", "m", "s", "f",
                        results=[CaseResult("x", "passed", 1.0, "ok")])
    path = write_report_md(batch, tmp_path, title="MBPP")
    assert "（无失败采样）" in path.read_text(encoding="utf-8")


def test_report_md_multi_sample_aggregation(tmp_path: Path) -> None:
    """多次采样报告：attempt 列 + k/N 聚合表 + flaky 提示。"""
    results = [
        CaseResult("stable", "passed", 1.0, "ok", attempt=1),
        CaseResult("stable", "passed", 1.0, "ok", attempt=2),
        CaseResult("wobbly", "passed", 1.0, "ok", attempt=1),
        CaseResult("wobbly", "failed", 0.0, "bad", attempt=2),
    ]
    batch = BatchResult("b3", "unit", "m", "s", "f", results=results)
    text = write_report_md(batch, tmp_path, title="Unit").read_text(encoding="utf-8")

    assert "| case | attempt | status |" in text
    assert "| wobbly | 2 | failed |" in text
    assert "## Case 级聚合（k/N 次采样通过）" in text
    assert "| stable | 2/2 | 稳定通过 |" in text
    assert "| wobbly | 1/2 | 不稳定 |" in text
    assert "不稳定 case 是 harness 迭代最值得排查的对象" in text
    assert "wobbly（第 2 次）" in text


def test_cases_file_roundtrip(tmp_path: Path) -> None:
    cases = [Case(case_id="c1", task="t", check="print(1)"),
             Case(case_id="c2", task="t2", expectations=[{"type": "no_tool_errors",
                                                          "weight": 1.0}])]
    save_cases(cases, tmp_path)
    assert load_cases(tmp_path) == cases


# ─── rebuild：对已存轨迹重评分（不重新执行 Agent）────────────────────────


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    """跑一个真实小批次（Mock 适配器），产物自包含。"""
    cases = [
        Case(case_id="passing", task="normal task",
             expectations=[{"type": "no_tool_errors", "weight": 1.0}]),
        Case(case_id="failing", task="this will fail",
             expectations=[{"type": "no_tool_errors", "weight": 1.0}]),
    ]
    batch_dir = new_batch_dir(tmp_path, "unit", 2, 2)
    save_cases(cases, batch_dir)

    async def drive() -> None:
        for case in cases:
            await run_case(case, MockAdapter(), model="mock", batch_dir=batch_dir,
                           timeout_s=5)

    import asyncio

    asyncio.run(drive())
    return batch_dir


async def test_rebuild_regrades_saved_traces(batch_dir: Path) -> None:
    """重评分与首跑结论一致（断言式离线重算）。"""
    batch = await rebuild(batch_dir)
    by_id = {r.case_id: r.status for r in batch.results}
    assert by_id == {"passing": "passed", "failing": "failed"}
    assert batch.pass_rate == 0.5


async def test_rebuild_reads_attempt_files_and_legacy_layout(tmp_path: Path) -> None:
    """rebuild 兼容两种布局：-a<attempt> 多采样与旧版无后缀单采样。"""
    from mcc_eval.events import LlmChunk, RunStarted, ToolResult

    cases = [Case(case_id="multi", task="t", expectations=[
        {"type": "no_tool_errors", "weight": 1.0}])]
    save_cases(cases, tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()

    def _write(name: str, *, tool_error: bool = False) -> None:
        with open(runs / name, "w", encoding="utf-8") as fh:
            fh.write(RunStarted(seq=0, run_id="r", case_id="multi", task="t",
                                model="m").model_dump_json() + "\n")
            fh.write(LlmChunk(seq=1, turn=0, delta="x").model_dump_json() + "\n")
            if tool_error:
                fh.write(ToolResult(seq=2, call_id="e", output="boom",
                                    is_error=True).model_dump_json() + "\n")
            fh.write(RunFinished(seq=3, status="passed", turns=1,
                                 cost_usd=0.01).model_dump_json() + "\n")

    _write("multi-a1.jsonl")
    _write("multi-a2.jsonl")
    _write("multi-a3.jsonl", tool_error=True)  # 前两次过、第三次挂 → flaky

    batch = await rebuild(tmp_path)
    assert [r.attempt for r in batch.results] == [1, 2, 3]
    assert batch.flaky_cases == 1
    assert batch.case_pass_counts() == [("multi", 2, 3)]

    # 旧布局：无后缀单文件视为 attempt 1
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    save_cases([Case(case_id="old", task="t")], legacy)
    (legacy / "runs").mkdir()
    _write_legacy = legacy / "runs" / "old.jsonl"
    _write_legacy.write_text(
        RunStarted(seq=0, run_id="r", case_id="old", task="t",
                   model="m").model_dump_json() + "\n"
        + RunFinished(seq=1, status="passed", turns=1).model_dump_json() + "\n",
        encoding="utf-8",
    )
    old_batch = await rebuild(legacy)
    assert old_batch.results[0].attempt == 1
    assert old_batch.results[0].status == "passed"


async def test_rebuild_missing_trace_is_error(tmp_path: Path) -> None:
    cases = [Case(case_id="ghost", task="t")]
    save_cases(cases, tmp_path)
    batch = await rebuild(tmp_path)
    assert batch.results[0].status == "error"
    assert "trace file missing" in batch.results[0].reason


async def test_rebuild_execution_terminal_preserved(tmp_path: Path) -> None:
    """轨迹末事件是 error / timeout 的，重评分保留终态不再评分。"""
    from mcc_eval.events import RunFinished

    cases = [Case(case_id="crashed", task="t")]
    save_cases(cases, tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "crashed.jsonl").write_text(
        RunFinished(seq=0, status="error").model_dump_json() + "\n", encoding="utf-8"
    )
    batch = await rebuild(tmp_path)
    assert batch.results[0].status == "error"
    assert "execution terminated as error" in batch.results[0].reason


async def test_rebuild_execution_check_reruns_on_work_dir(tmp_path: Path) -> None:
    """执行式重评分：在保留的 work/ 目录上重跑官方测试。"""
    cases = [Case(case_id="exec", task="t",
                  check="from solution import double as candidate\nassert candidate(3) == 6\n")]
    save_cases(cases, tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    work = tmp_path / "work" / "exec"
    work.mkdir(parents=True)
    (work / "solution.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    from mcc_eval.events import LlmChunk, RunStarted

    with open(runs / "exec.jsonl", "w", encoding="utf-8") as fh:
        fh.write(RunStarted(seq=0, run_id="r", case_id="exec", task="t",
                            model="m").model_dump_json() + "\n")
        fh.write(LlmChunk(seq=1, turn=0, delta="PASSED").model_dump_json() + "\n")
        fh.write(RunFinished(seq=2, status="passed", cost_usd=0.01,
                             turns=1).model_dump_json() + "\n")

    batch = await rebuild(tmp_path)
    assert batch.results[0].status == "passed"


def test_load_trace_events_validates_schema(tmp_path: Path) -> None:
    from mcc_eval.events import LlmChunk

    path = tmp_path / "t.jsonl"
    path.write_text(
        LlmChunk(seq=0, turn=0, delta="hi").model_dump_json() + "\n"
        + json.dumps({"type": "bogus"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_trace_events(path)
