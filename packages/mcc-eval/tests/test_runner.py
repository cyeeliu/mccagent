"""执行器测试：run_case 全链路、多次采样、setup 预置、超时 / 异常路径。"""

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
    ProgressTracker,
    apply_setup,
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


class _FlakyAdapter:
    """第一次跑挂、之后全过——验证多次采样能抓出不稳定 case。"""

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        from mcc_eval.events import RunStarted

        self.calls += 1
        yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                         task=req.task, model=req.model)
        if self.calls == 1:
            raise RuntimeError("first attempt flaked")
        yield RunFinished(seq=1, status="passed")


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
    assert result.attempt == 1

    trace = batch_dir / "runs" / "case-1-a1.jsonl"
    lines = [json.loads(ln) for ln in trace.read_text(encoding="utf-8").splitlines() if ln]
    assert lines[0]["type"] == "run_started"
    assert lines[-1]["type"] == "run_finished"
    assert (batch_dir / "work" / "case-1-a1").is_dir()


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
             (batch_dir / "runs" / "case-1-a1.jsonl").read_text(encoding="utf-8")
             .splitlines() if ln]
    assert lines[-1]["type"] == "run_finished"
    assert lines[-1]["status"] == "timeout"


async def test_run_case_adapter_crash_reports_error(tmp_path: Path) -> None:
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(), _ExplodingAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5)
    assert result.status == "error"
    assert "adapter exploded" in result.reason
    assert result.score is None


async def test_trace_file_flushed_per_event(tmp_path: Path) -> None:
    """轨迹逐事件即时落盘：运行中（case 未结束时）外部读盘即可见。

    利用 async generator 的挂起点：第二条事件 yield 前，第一条
    必须已经写盘并 flush——不依赖 sleep 的确定性验证。
    """
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    trace_path = batch_dir / "runs" / "case-1-a1.jsonl"
    on_disk: list[str] = []

    class _GatedAdapter:
        async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
            from mcc_eval.events import LlmChunk, RunStarted

            yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                             task=req.task, model=req.model)
            on_disk.append(trace_path.read_text(encoding="utf-8"))
            yield LlmChunk(seq=1, turn=0, delta="working")
            on_disk.append(trace_path.read_text(encoding="utf-8"))
            yield RunFinished(seq=2, status="passed", turns=1)

    result = await run_case(_case(), _GatedAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5)
    assert result.status == "passed"
    assert '"type": "run_started"' in on_disk[0]  # 首条已实时可见
    assert '"type": "llm_chunk"' in on_disk[1] and '"run_finished"' not in on_disk[1]


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
    assert batch.stable_cases == 3
    assert batch.flaky_cases == 0

    data = batch.to_dict()
    assert data["summary"]["total"] == 4
    assert data["summary"]["cases"] == 4
    assert len(data["results"]) == 4


async def test_run_batch_progress_lines(tmp_path: Path, capsys) -> None:
    cases = [_case(case_id="only")]
    batch_dir = new_batch_dir(tmp_path, "unit", 1, 1)
    await run_batch(cases, MockAdapter(), dataset="unit", model="mock",
                    batch_dir=batch_dir, concurrency=1, timeout_s=5)
    err = capsys.readouterr().err
    assert "[1/1] only:" in err


# ─── 多次采样 ────────────────────────────────────────────────────────────


async def test_run_batch_multiple_samples_exposes_flaky(tmp_path: Path) -> None:
    """首采崩溃、重采通过 → flaky 统计命中，k/N 为 1/2。"""
    adapter = _FlakyAdapter()
    cases = [_case(case_id="flaky")]
    batch_dir = new_batch_dir(tmp_path, "unit", 1, 1)
    batch = await run_batch(cases, adapter, dataset="unit", model="mock",
                            batch_dir=batch_dir, timeout_s=5, runs_per_case=2)

    assert batch.total == 2
    statuses = sorted(r.status for r in batch.results)
    assert statuses == ["error", "passed"]
    assert batch.case_pass_counts() == [("flaky", 1, 2)]
    assert batch.flaky_cases == 1
    assert batch.stable_cases == 0
    assert (batch_dir / "runs" / "flaky-a1.jsonl").exists()
    assert (batch_dir / "runs" / "flaky-a2.jsonl").exists()


async def test_run_batch_progress_counts_total_runs(tmp_path: Path, capsys) -> None:
    cases = [_case(case_id="x")]
    batch_dir = new_batch_dir(tmp_path, "unit", 1, 1)
    await run_batch(cases, MockAdapter(), dataset="unit", model="mock",
                    batch_dir=batch_dir, timeout_s=5, runs_per_case=3)
    err = capsys.readouterr().err
    assert "[3/3] x#:" in err


# ─── setup 环境预置 ──────────────────────────────────────────────────────


async def test_run_case_setup_files_preset(tmp_path: Path) -> None:
    """setup 映射在 Agent 开跑前写入工作目录（嵌套路径按需创建）。"""
    case = _case(case_id="setup-1", setup={
        "src/math.py": "def divide(a, b):\n    return a / b\n",
        "tests/test_math.py": "from src.math import divide\n",
    })
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)

    seen: dict[str, str] = {}

    class _ReaderAdapter:
        async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
            from mcc_eval.events import LlmChunk

            root = Path(req.work_dir or ".")
            seen["math"] = (root / "src" / "math.py").read_text(encoding="utf-8")
            seen["test"] = (root / "tests" / "test_math.py").read_text(encoding="utf-8")
            yield LlmChunk(seq=0, turn=0, delta="read preset files")
            yield RunFinished(seq=1, status="passed", turns=1)

    result = await run_case(case, _ReaderAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5)
    assert result.status == "passed"
    assert "divide" in seen["math"]
    assert "src.math" in seen["test"]


@pytest.mark.parametrize("bad", ["../escape.py", "a/../../b.txt"])
def test_apply_setup_rejects_path_escape(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError, match="escapes work dir"):
        apply_setup(tmp_path, {bad: "content"})


def test_apply_setup_writes_nested_files(tmp_path: Path) -> None:
    apply_setup(tmp_path, {"a/b/c.txt": "hello", "top.py": "x = 1\n"})
    assert (tmp_path / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "hello"
    assert (tmp_path / "top.py").read_text(encoding="utf-8") == "x = 1\n"


# ─── adapter 自报终态 ────────────────────────────────────────────────────


async def test_run_case_respects_adapter_terminal_error(tmp_path: Path) -> None:
    """adapter 报 error（如 LLM 认证失败）时保留终态不评分，reason 透传。"""

    class _AuthFailAdapter:
        async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
            from mcc_eval.events import RunStarted

            yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                             task=req.task, model=req.model)
            yield RunFinished(seq=1, status="error", turns=0, cost_usd=0.0,
                              reason="no ANTHROPIC_API_KEY in environment")

    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(check="assert True\n"), _AuthFailAdapter(),
                            model="mock", batch_dir=batch_dir, timeout_s=5)
    assert result.status == "error"
    assert result.score is None
    assert result.reason == "no ANTHROPIC_API_KEY in environment"
    lines = [json.loads(ln) for ln in
             (batch_dir / "runs" / "case-1-a1.jsonl").read_text(encoding="utf-8")
             .splitlines() if ln]
    assert lines[-1]["reason"] == "no ANTHROPIC_API_KEY in environment"


async def test_run_case_adapter_terminal_without_reason_falls_back(
    tmp_path: Path,
) -> None:
    """旧 adapter 无 reason 时兜底文案仍可读。"""

    class _BareErrorAdapter:
        async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
            from mcc_eval.events import RunStarted

            yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                             task=req.task, model=req.model)
            yield RunFinished(seq=1, status="error")

    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    result = await run_case(_case(), _BareErrorAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5)
    assert result.status == "error"
    assert result.reason == "execution terminated as error"


# ─── progress.json 实时进度 ──────────────────────────────────────────────


async def test_progress_json_updates_per_loop(tmp_path: Path) -> None:
    """Agent 每推进一轮 loop，progress.json 即重写（case 未结束时可见）。

    利用 async generator 挂起点做确定性验证：两个 yield 之间读盘，
    断言 loop 计数与事件类型已实时落盘。
    """
    batch_dir = new_batch_dir(tmp_path, "unit", 0, 1)
    progress_path = batch_dir / "progress.json"
    tracker = ProgressTracker(batch_dir, dataset="unit", model="mock",
                              total_runs=1, started_at="2026-10-20T10:00:00+00:00")
    snaps: list[dict] = []

    class _LoopGatedAdapter:
        async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
            from mcc_eval.events import LlmChunk, RunStarted, ToolCall, ToolResult

            yield RunStarted(seq=0, run_id=req.run_id, case_id=req.case_id,
                             task=req.task, model=req.model)
            snaps.append(json.loads(progress_path.read_text(encoding="utf-8")))
            yield LlmChunk(seq=1, turn=0, delta="thinking")
            yield ToolCall(seq=2, turn=1, call_id="t1", tool_name="read_file",
                           args={"path": "a.py"})
            snaps.append(json.loads(progress_path.read_text(encoding="utf-8")))
            yield ToolResult(seq=3, call_id="t1", output="ok")
            yield LlmChunk(seq=4, turn=2, delta="done")
            snaps.append(json.loads(progress_path.read_text(encoding="utf-8")))
            yield RunFinished(seq=5, status="passed", turns=3)

    result = await run_case(_case(), _LoopGatedAdapter(), model="mock",
                            batch_dir=batch_dir, timeout_s=5, progress=tracker)
    assert result.status == "passed"

    assert snaps[0]["active"] == [{
        "case_id": "case-1", "attempt": 1, "loop": 0, "events": 0,
        "last_event": None, "last_event_at": None,
    }]
    assert snaps[1]["active"][0]["loop"] == 2  # turn=1 → 第 2 轮 loop
    assert snaps[1]["active"][0]["last_event"] == "tool_call"
    assert snaps[1]["active"][0]["events"] == 3
    assert snaps[2]["active"][0]["loop"] == 3  # turn=2 → 第 3 轮
    assert snaps[2]["active"][0]["last_event"] == "llm_chunk"
    assert snaps[2]["active"][0]["events"] == 5
    assert all(s["running"] is True for s in snaps)


async def test_progress_json_batch_lifecycle(tmp_path: Path) -> None:
    """批次跑完后 progress.json 收敛为终态：active 清空、completed 齐全。"""
    cases = [_case(case_id="ok"), _case(case_id="bad", task="this will fail")]
    batch_dir = new_batch_dir(tmp_path, "unit", 2, 2)
    batch = await run_batch(cases, MockAdapter(), dataset="unit", model="mock",
                            batch_dir=batch_dir, concurrency=2, timeout_s=5,
                            runs_per_case=2)
    assert batch.passed == 2 and batch.failed == 2

    data = json.loads((batch_dir / "progress.json").read_text(encoding="utf-8"))
    assert data["running"] is False
    assert data["total_runs"] == 4
    assert data["done_runs"] == 4
    assert data["active"] == []
    assert len(data["completed"]) == 4
    by_key = {(c["case_id"], c["attempt"]): c["status"] for c in data["completed"]}
    assert by_key == {
        ("ok", 1): "passed", ("ok", 2): "passed",
        ("bad", 1): "failed", ("bad", 2): "failed",
    }
    assert all(c["reason"] for c in data["completed"])


async def test_progress_json_survives_crash_cleanup(tmp_path: Path) -> None:
    """adapter 崩溃的采样也从 active 清理、按 error 记入 completed。"""
    batch_dir = new_batch_dir(tmp_path, "unit", 1, 1)
    await run_batch([_case()], _ExplodingAdapter(), dataset="unit", model="mock",
                    batch_dir=batch_dir, timeout_s=5)

    data = json.loads((batch_dir / "progress.json").read_text(encoding="utf-8"))
    assert data["done_runs"] == 1
    assert data["active"] == []
    assert data["completed"][0]["status"] == "error"
    assert "adapter exploded" in data["completed"][0]["reason"]


# ─── snapshot 折叠 ──────────────────────────────────────────────────────


def test_snapshot_from_events_folds_pipeline() -> None:
    from mcc_eval.events import LlmChunk, ToolCall, ToolResult

    events = [
        ToolCall(seq=0, turn=0, call_id="a", tool_name="bash", args="pytest"),
        ToolResult(seq=1, call_id="a", output="boom", is_error=True),
        LlmChunk(seq=2, turn=0, delta="hello "),
        LlmChunk(seq=3, turn=0, delta="world"),
        RunFinished(seq=4, status="passed", cost_usd=0.25, turns=7),
    ]
    snap = snapshot_from_events(events)
    assert len(snap.tool_calls) == 1
    assert snap.had_error
    assert "".join(snap.text_outputs) == "hello world"
    assert snap.cost_usd == 0.25
    assert snap.turns == 7


def test_case_result_and_batch_result_dicts() -> None:
    r = CaseResult(case_id="x", status="failed", score=0.25, reason="why",
                   cost_usd=0.01, turns=2, duration_s=1.5, details=["✗ a"],
                   attempt=3)
    d = r.to_dict()
    assert d["duration_s"] == 1.5 and d["details"] == ["✗ a"]
    assert d["attempt"] == 3

    b = BatchResult(batch_id="b", dataset="d", model="m", started_at="s",
                    finished_at="f", results=[r])
    summary = b.to_dict()["summary"]
    assert summary["pass_rate"] == 0.0
    assert summary["cases"] == 1
    assert summary["flaky_cases"] == 0
