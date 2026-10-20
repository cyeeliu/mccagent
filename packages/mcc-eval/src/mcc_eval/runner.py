"""执行器：单进程 asyncio 并发跑一批 Case，轨迹与产物落盘。

与分布式架构（server + runner + 租约 + 心跳 + reaper）的关键区别：
状态只存在于一个进程的内存与文件系统里，没有跨进程副本就没有
不一致。每个 Case 的每次采样（attempt）的产物是三件套：

  - runs/<case_id>-a<attempt>.jsonl   完整轨迹（六类事件，流式落盘）
  - work/<case_id>-a<attempt>/        Agent 工作目录（保留现场，供
                                      执行式评分与人工检查）
  - results.json 里的记录             状态、分数、理由、采样序号

支持每 case 多次采样（runs_per_case）：Agent 输出非确定，单次
pass/fail 没有信息量，case 级结论看 k/N 聚合（全过 / 部分 flaky /
全挂）。

评分状态与执行状态解耦：执行层崩溃 / 超时才报 error / timeout，
Agent 执行完成的一律交给评分器判定 passed / failed——「过程有报错」
不再被误升为「结果 error」。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcc_eval.adapters import AgentAdapter, RunRequest
from mcc_eval.datasets import Case
from mcc_eval.events import (
    LlmChunk,
    RunFinished,
    ToolCall,
    ToolResult,
    TraceEvent,
)
from mcc_eval.scoring import TraceSnapshot, score_case


@dataclass
class CaseResult:
    """单个 Case 单次采样的最终结论（results.json 中的一条记录）。

    attempt 是同一 case 的第几次采样（从 1 起）；非确定性下
    单次 pass/fail 没有信息量，结论要看 case 级的 k/N 聚合。
    """

    case_id: str
    status: str  # passed | failed | error | timeout
    score: float | None
    reason: str
    cost_usd: float | None = None
    turns: int = 0
    duration_s: float = 0.0
    details: list[str] = field(default_factory=list)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "attempt": self.attempt,
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
            "cost_usd": self.cost_usd,
            "turns": self.turns,
            "duration_s": round(self.duration_s, 3),
            "details": self.details,
        }


@dataclass
class BatchResult:
    """一批 Case 的汇总（results.json 的顶层结构）。"""

    batch_id: str
    dataset: str
    model: str
    started_at: str
    finished_at: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "failed")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status in ("error", "timeout"))

    @property
    def cost_usd(self) -> float:
        return round(sum(r.cost_usd or 0.0 for r in self.results), 4)

    @property
    def pass_rate(self) -> float:
        scored = self.passed + self.failed
        return round(self.passed / scored, 4) if scored else 0.0

    @property
    def stable_cases(self) -> int:
        """全部采样一致通过的 case 数（k == N）。"""
        return sum(1 for _, k, n in self.case_pass_counts() if n and k == n)

    @property
    def flaky_cases(self) -> int:
        """部分采样通过的 case 数（0 < k < N）——不稳定，最值得排查。"""
        return sum(1 for _, k, n in self.case_pass_counts() if 0 < k < n)

    def case_pass_counts(self) -> list[tuple[str, int, int]]:
        """每 case 的 (case_id, 通过次数, 总采样次数)，按 case_id 排序。"""
        counts: dict[str, list[int]] = {}
        for r in self.results:
            pair = counts.setdefault(r.case_id, [0, 0])
            pair[1] += 1
            if r.status == "passed":
                pair[0] += 1
        return [(cid, counts[cid][0], counts[cid][1]) for cid in sorted(counts)]

    def to_dict(self) -> dict[str, Any]:
        counts = self.case_pass_counts()
        return {
            "batch_id": self.batch_id,
            "dataset": self.dataset,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "pass_rate": self.pass_rate,
                "cost_usd": self.cost_usd,
                "cases": len(counts),
                "stable_cases": self.stable_cases,
                "flaky_cases": self.flaky_cases,
            },
            "results": [r.to_dict() for r in self.results],
        }


def new_batch_dir(out_root: Path, dataset: str, limit: int, total: int) -> Path:
    """创建带时间戳的批次目录：<时间戳>-<dataset>-<数量>。"""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    batch_dir = out_root / f"{stamp}-{dataset}-{limit or total}"
    (batch_dir / "runs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "work").mkdir(parents=True, exist_ok=True)
    return batch_dir


def snapshot_from_events(events: list[TraceEvent]) -> TraceSnapshot:
    """把轨迹事件折叠为评分用的快照（含末态 cost / turns）。"""
    snap = TraceSnapshot()
    for ev in events:
        if isinstance(ev, ToolCall):
            snap.tool_calls.append(ev)
        elif isinstance(ev, ToolResult):
            snap.tool_results.append(ev)
            snap.had_error = snap.had_error or ev.is_error
        elif isinstance(ev, LlmChunk):
            snap.text_outputs.append(ev.delta)
        elif isinstance(ev, RunFinished):
            snap.cost_usd = ev.cost_usd or 0.0
            snap.turns = ev.turns
    return snap


def apply_setup(work_dir: Path, setup: dict[str, str]) -> None:
    """按声明式映射预置工作目录：``{相对路径: 文件内容}``。

    路径限定在工作目录内（拒绝 ``..`` 逃逸），父目录按需创建。
    """
    root = work_dir.resolve()
    for rel, content in setup.items():
        target = (work_dir / rel).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"setup path escapes work dir: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _dump_event(fh: Any, ev: TraceEvent) -> None:
    fh.write(json.dumps(ev.model_dump(mode="json"), ensure_ascii=False) + "\n")


async def run_case(
    case: Case,
    adapter: AgentAdapter,
    *,
    model: str,
    batch_dir: Path,
    max_turns: int = 20,
    timeout_s: float = 600.0,
    attempt: int = 1,
) -> CaseResult:
    """执行单个 Case 的一次采样：预置环境 → 轨迹流式落盘 → 评分。

    每次采样独立的轨迹文件与工作目录（``-a<attempt>`` 后缀），
    采样之间互不污染。超时 / 执行异常时补一条 RunFinished 落盘，
    保证轨迹文件完整（末事件永远标记终态）；error / timeout
    永远附 reason。
    """
    started = time.monotonic()
    suffix = f"-a{attempt}"
    trace_path = batch_dir / "runs" / f"{case.case_id}{suffix}.jsonl"
    work_dir = batch_dir / "work" / f"{case.case_id}{suffix}"
    work_dir.mkdir(parents=True, exist_ok=True)
    apply_setup(work_dir, case.setup)

    req = RunRequest(
        run_id=f"{case.case_id}{suffix}-{int(time.time() * 1000)}",
        case_id=case.case_id,
        task=case.task,
        model=model,
        max_turns=max_turns,
        work_dir=str(work_dir),
    )

    events: list[TraceEvent] = []
    terminal = "done"
    error_reason: str | None = None
    with open(trace_path, "w", encoding="utf-8") as fh:

        async def consume() -> None:
            async for ev in adapter.run(req):
                events.append(ev)
                _dump_event(fh, ev)

        try:
            await asyncio.wait_for(consume(), timeout=timeout_s)
        except TimeoutError:
            terminal = "timeout"
            error_reason = f"run exceeded timeout of {timeout_s}s"
        except Exception as exc:
            terminal = "error"
            error_reason = f"{type(exc).__name__}: {exc}"

        if terminal != "done":
            finished = RunFinished(
                seq=len(events), status=terminal, score=None, cost_usd=None, turns=0,
            )
            events.append(finished)
            _dump_event(fh, finished)
        fh.flush()

    duration = time.monotonic() - started

    if terminal != "done":
        return CaseResult(
            case_id=case.case_id,
            status=terminal,
            score=None,
            reason=error_reason or terminal,
            duration_s=duration,
            attempt=attempt,
        )

    run_finished = next(
        (ev for ev in reversed(events) if isinstance(ev, RunFinished)), None
    )
    snap = snapshot_from_events(events)
    scored = score_case(
        snap, check=case.check, expectations=case.expectations, work_dir=work_dir
    )
    return CaseResult(
        case_id=case.case_id,
        status=scored.status,
        score=scored.score,
        reason=scored.reason,
        cost_usd=run_finished.cost_usd if run_finished else None,
        turns=run_finished.turns if run_finished else 0,
        duration_s=duration,
        details=scored.details,
        attempt=attempt,
    )


async def run_batch(
    cases: list[Case],
    adapter: AgentAdapter,
    *,
    dataset: str,
    model: str,
    batch_dir: Path,
    concurrency: int = 2,
    max_turns: int = 20,
    timeout_s: float = 600.0,
    runs_per_case: int = 1,
) -> BatchResult:
    """并发执行一批 Case，每 case 采样 ``runs_per_case`` 次。

    非确定性下同一任务多次运行结果可能不同：单次 pass/fail 没有
    信息量，多次采样暴露不稳定 case（flaky）。进度逐条打到 stderr，
    总数 = case 数 × 采样数。
    """
    started_at = datetime.now(UTC).isoformat()
    sem = asyncio.Semaphore(max(1, concurrency))
    total_runs = len(cases) * max(1, runs_per_case)
    done_count = 0

    async def one(case: Case, attempt: int) -> CaseResult:
        nonlocal done_count
        async with sem:
            result = await run_case(
                case,
                adapter,
                model=model,
                batch_dir=batch_dir,
                max_turns=max_turns,
                timeout_s=timeout_s,
                attempt=attempt,
            )
            done_count += 1
            print(
                f"[{done_count}/{total_runs}] {result.case_id}"
                f"{'#' if attempt > 1 else ''}: "
                f"{result.status} (score={result.score})",
                file=sys.stderr,
            )
            return result

    jobs = [
        one(case, attempt)
        for case in cases
        for attempt in range(1, max(1, runs_per_case) + 1)
    ]
    results = await asyncio.gather(*jobs)
    return BatchResult(
        batch_id=batch_dir.name,
        dataset=dataset,
        model=model,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        results=list(results),
    )
