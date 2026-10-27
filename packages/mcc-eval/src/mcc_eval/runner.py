"""执行器：单进程 asyncio 并发跑一批 Case，轨迹与产物落盘。

与分布式架构（server + runner + 租约 + 心跳 + reaper）的关键区别：
状态只存在于一个进程的内存与文件系统里，没有跨进程副本就没有
不一致。每个 Case 的每次采样（attempt）的产物是三件套：

  - runs/<case_id>-a<attempt>.jsonl   完整轨迹（六类事件，逐事件落盘）
  - work/<case_id>-a<attempt>/        Agent 工作目录（保留现场，供
                                      执行式评分与人工检查）
  - results.json 里的记录             状态、分数、理由、采样序号

批次目录另有 progress.json：实时进度镜像——Agent 每推进一轮 loop
（轨迹事件 turn 变化）或发生 case 状态迁移即原子重写，运行中读它
即可监控，不必等批次结束的 results.json。

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
import os
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
from mcc_eval.judge import JudgeScorer
from mcc_eval.scoring import TraceSnapshot, score_case

PROGRESS_FILE = "progress.json"


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


class ProgressTracker:
    """批次实时进度镜像：每次状态迁移原子重写 progress.json。

    写盘时机与「每跑一个 loop 就实时更新」对齐：

    - case 开始 / 完成（状态迁移）
    - 轨迹事件的 ``turn`` 变化——Agent 推进了新一轮 loop

    原子写（临时文件 + os.replace）保证外部任意时刻读到的都是
    完整 JSON；文件很小（每个采样一条 active/completed 记录），
    重写开销可忽略。所有方法都在同一事件循环里被调用，无需加锁。
    """

    def __init__(
        self,
        batch_dir: Path,
        *,
        dataset: str,
        model: str,
        total_runs: int,
        started_at: str,
    ) -> None:
        self._path = batch_dir / PROGRESS_FILE
        self._state: dict[str, Any] = {
            "batch_id": batch_dir.name,
            "dataset": dataset,
            "model": model,
            "started_at": started_at,
            "updated_at": started_at,
            "running": True,
            "total_runs": total_runs,
            "done_runs": 0,
            "active": [],
            "completed": [],
        }
        self._by_key: dict[tuple[str, int], dict[str, Any]] = {}

    def case_started(self, case_id: str, attempt: int) -> None:
        entry: dict[str, Any] = {
            "case_id": case_id,
            "attempt": attempt,
            "loop": 0,
            "events": 0,
            "last_event": None,
            "last_event_at": None,
        }
        self._state["active"].append(entry)
        self._by_key[(case_id, attempt)] = entry
        self._write()

    def observe(
        self, case_id: str, attempt: int, ev: TraceEvent, event_count: int
    ) -> None:
        """吸收一个轨迹事件：内存常更新，loop 推进时写盘。"""
        entry = self._by_key.get((case_id, attempt))
        if entry is None:
            return
        entry["events"] = event_count
        entry["last_event"] = getattr(ev, "type", None)
        entry["last_event_at"] = getattr(ev, "ts", None)
        turn = getattr(ev, "turn", None)
        if turn is not None and turn + 1 != entry["loop"]:
            entry["loop"] = turn + 1
            self._write()

    def case_finished(self, result: CaseResult) -> None:
        entry = self._by_key.pop((result.case_id, result.attempt), None)
        if entry is not None:
            self._state["active"].remove(entry)
        self._state["completed"].append(
            {
                "case_id": result.case_id,
                "attempt": result.attempt,
                "status": result.status,
                "score": result.score,
                "reason": result.reason,
                "cost_usd": result.cost_usd,
                "turns": result.turns,
                "duration_s": round(result.duration_s, 3),
            }
        )
        self._state["done_runs"] += 1
        self._write()

    def finish(self) -> None:
        self._state["running"] = False
        self._write()

    def _write(self) -> None:
        self._state["updated_at"] = datetime.now(UTC).isoformat()
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)


def _dump_event(fh: Any, ev: TraceEvent) -> None:
    """逐事件落盘并即时 flush。

    只写不 flush 的话事件停留在解释器缓冲区里，运行中 tail 不可见、
    进程被杀时已产生的事件全部丢失——轨迹的完整性承诺就没了。
    每条一次 flush 的开销在事件频率（远低于 token 速率）下可忽略。
    """
    fh.write(json.dumps(ev.model_dump(mode="json"), ensure_ascii=False) + "\n")
    fh.flush()


async def run_case(
    case: Case,
    adapter: AgentAdapter,
    *,
    model: str,
    batch_dir: Path,
    max_turns: int = 40,
    timeout_s: float = 600.0,
    attempt: int = 1,
    progress: ProgressTracker | None = None,
    judge: JudgeScorer | None = None,
) -> CaseResult:
    """执行单个 Case 的一次采样：预置环境 → 轨迹流式落盘 → 评分。

    每次采样独立的轨迹文件与工作目录（``-a<attempt>`` 后缀），
    采样之间互不污染。超时 / 执行异常时补一条 RunFinished 落盘，
    保证轨迹文件完整（末事件永远标记终态）；error / timeout
    永远附 reason。progress 存在时，case 开始与每轮 loop 推进
    都会实时反映到 progress.json。
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

    if progress is not None:
        progress.case_started(case.case_id, attempt)

    events: list[TraceEvent] = []
    terminal = "done"
    error_reason: str | None = None
    with open(trace_path, "w", encoding="utf-8") as fh:

        async def consume() -> None:
            async for ev in adapter.run(req):
                events.append(ev)
                _dump_event(fh, ev)
                if progress is not None:
                    progress.observe(case.case_id, attempt, ev, len(events))

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
    # adapter 自报的执行终态优先于评分器（与 rebuild 语义一致）：
    # error / timeout 说明 Agent 根本没跑完（如 LLM 认证失败、网关
    # 不可达），对残缺轨迹评分没有意义，reason 透传供排障
    if run_finished is not None and run_finished.status in ("error", "timeout"):
        return CaseResult(
            case_id=case.case_id,
            status=run_finished.status,
            score=None,
            reason=run_finished.reason
            or f"execution terminated as {run_finished.status}",
            cost_usd=run_finished.cost_usd,
            turns=run_finished.turns,
            duration_s=duration,
            attempt=attempt,
        )

    snap = snapshot_from_events(events)
    judge_results = None
    if judge is not None:
        judge_results = await judge.prescore(
            " ".join(snap.text_outputs), case.expectations
        )
    scored = score_case(
        snap, check=case.check, expectations=case.expectations,
        work_dir=work_dir, judge_results=judge_results,
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
    max_turns: int = 40,
    timeout_s: float = 600.0,
    runs_per_case: int = 1,
    judge: JudgeScorer | None = None,
) -> BatchResult:
    """并发执行一批 Case，每 case 采样 ``runs_per_case`` 次。

    非确定性下同一任务多次运行结果可能不同：单次 pass/fail 没有
    信息量，多次采样暴露不稳定 case（flaky）。进度逐条打到 stderr，
    总数 = case 数 × 采样数；progress.json 随每轮 loop 与每次
    case 状态迁移实时更新。
    """
    started_at = datetime.now(UTC).isoformat()
    sem = asyncio.Semaphore(max(1, concurrency))
    total_runs = len(cases) * max(1, runs_per_case)
    done_count = 0
    tracker = ProgressTracker(
        batch_dir,
        dataset=dataset,
        model=model,
        total_runs=total_runs,
        started_at=started_at,
    )

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
                progress=tracker,
                judge=judge,
            )
            tracker.case_finished(result)
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
    tracker.finish()
    return BatchResult(
        batch_id=batch_dir.name,
        dataset=dataset,
        model=model,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        results=list(results),
    )
