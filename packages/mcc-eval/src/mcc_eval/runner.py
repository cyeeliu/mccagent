"""执行器：单进程 asyncio 并发跑一批 Case，轨迹与产物落盘。

与分布式架构（server + runner + 租约 + 心跳 + reaper）的关键区别：
状态只存在于一个进程的内存与文件系统里，没有跨进程副本就没有
不一致。每个 Case 的产物是三件套：

  - runs/<case_id>.jsonl   完整轨迹（六类事件，流式落盘）
  - work/<case_id>/        Agent 工作目录（保留现场，供执行式评分与人工检查）
  - results.json 里的记录   状态、分数、理由

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
    """单个 Case 的最终结论（results.json 中的一条记录）。"""

    case_id: str
    status: str  # passed | failed | error | timeout
    score: float | None
    reason: str
    cost_usd: float | None = None
    turns: int = 0
    duration_s: float = 0.0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
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

    def to_dict(self) -> dict[str, Any]:
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
    """把轨迹事件折叠为评分用的快照。"""
    snap = TraceSnapshot()
    for ev in events:
        if isinstance(ev, ToolCall):
            snap.tool_calls.append(ev)
        elif isinstance(ev, ToolResult):
            snap.tool_results.append(ev)
            snap.had_error = snap.had_error or ev.is_error
        elif isinstance(ev, LlmChunk):
            snap.text_outputs.append(ev.delta)
    return snap


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
) -> CaseResult:
    """执行单个 Case：轨迹流式落盘 → 评分 → 结论。

    超时 / 执行异常时补一条 RunFinished 落盘，保证轨迹文件完整
    （末事件永远标记终态）；error / timeout 永远附 reason。
    """
    started = time.monotonic()
    trace_path = batch_dir / "runs" / f"{case.case_id}.jsonl"
    work_dir = batch_dir / "work" / case.case_id
    work_dir.mkdir(parents=True, exist_ok=True)

    req = RunRequest(
        run_id=f"{case.case_id}-{int(time.time() * 1000)}",
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
) -> BatchResult:
    """并发执行一批 Case；进度逐条打到 stderr。"""
    started_at = datetime.now(UTC).isoformat()
    sem = asyncio.Semaphore(max(1, concurrency))
    done_count = 0

    async def one(case: Case) -> CaseResult:
        nonlocal done_count
        async with sem:
            result = await run_case(
                case,
                adapter,
                model=model,
                batch_dir=batch_dir,
                max_turns=max_turns,
                timeout_s=timeout_s,
            )
            done_count += 1
            print(
                f"[{done_count}/{len(cases)}] {result.case_id}: "
                f"{result.status} (score={result.score})",
                file=sys.stderr,
            )
            return result

    results = await asyncio.gather(*(one(c) for c in cases))
    return BatchResult(
        batch_id=batch_dir.name,
        dataset=dataset,
        model=model,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        results=list(results),
    )
