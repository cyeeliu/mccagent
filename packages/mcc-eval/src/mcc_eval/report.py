"""报告：results.json（机器读）+ report.md（人读），以及对已存日志重评分。

``rebuild`` 对齐 SWE-bench ``report`` 子命令的语义：对落盘的轨迹与
产物重新评分，不需要重新执行 Agent——断言式离线重算，执行式在保留的
work/ 目录上重跑官方测试。批次目录因此自包含（cases.json + 轨迹 +
产物），随时可以重新验证或事后换阈值重判。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcc_eval.datasets import Case
from mcc_eval.events import RunFinished, TraceEvent, trace_event_adapter
from mcc_eval.runner import BatchResult, CaseResult, snapshot_from_events
from mcc_eval.scoring import score_case

CASES_FILE = "cases.json"
RESULTS_FILE = "results.json"
REPORT_FILE = "report.md"


def save_cases(cases: list[Case], batch_dir: Path) -> None:
    """把批次用到的 Case 定义落盘（rebuild 的评分依据）。"""
    with open(batch_dir / CASES_FILE, "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in cases], fh, ensure_ascii=False, indent=2)


def load_cases(batch_dir: Path) -> list[Case]:
    with open(batch_dir / CASES_FILE, encoding="utf-8") as fh:
        return [Case.from_dict(item) for item in json.load(fh)]


def write_results_json(batch: BatchResult, batch_dir: Path) -> Path:
    path = batch_dir / RESULTS_FILE
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(batch.to_dict(), fh, ensure_ascii=False, indent=2)
    return path


def write_report_md(batch: BatchResult, batch_dir: Path, *, title: str) -> Path:
    """生成 markdown 报告：汇总 + 按采样明细 + case 级 k/N 聚合 + 失败理由。"""
    multi = any(r.attempt > 1 for r in batch.results)
    counts = batch.case_pass_counts()
    flaky = [(cid, k, n) for cid, k, n in counts if 0 < k < n]

    lines = [
        f"# 评测报告：{batch.batch_id}",
        "",
        f"- 数据集：{title}（{batch.dataset}）",
        f"- 模型：{batch.model}",
        f"- 时间：{batch.started_at} → {batch.finished_at}",
        f"- 通过率：**{batch.pass_rate:.1%}**（passed {batch.passed} / "
        f"failed {batch.failed} / error+timeout {batch.errors}，共 {batch.total}）",
        f"- 总成本：${batch.cost_usd:.4f}",
    ]
    if multi:
        per_case = max((n for _, _, n in counts), default=1)
        lines += [
            f"- 采样：每 case {per_case} 次——"
            f"稳定通过 {batch.stable_cases}，不稳定（flaky）{batch.flaky_cases}，"
            f"共 {len(counts)} 个 case",
        ]
    lines += ["", "## 明细", ""]
    if multi:
        lines += [
            "| case | attempt | status | score | turns | cost | duration |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    else:
        lines += [
            "| case | status | score | turns | cost | duration |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    for r in batch.results:
        score = "-" if r.score is None else f"{r.score:.2f}"
        cost = "-" if r.cost_usd is None else f"${r.cost_usd:.4f}"
        cols = [r.case_id, str(r.attempt) if multi else r.status]
        if multi:
            cols.append(r.status)
        cols += [score, str(r.turns), cost, f"{r.duration_s:.1f}s"]
        lines.append("| " + " | ".join(cols) + " |")

    if multi:
        lines += [
            "",
            "## Case 级聚合（k/N 次采样通过）",
            "",
            "| case | k/N | 判定 |",
            "| --- | --- | --- |",
        ]
        for cid, k, n in counts:
            verdict = "稳定通过" if k == n else ("不稳定" if k > 0 else "稳定失败")
            lines.append(f"| {cid} | {k}/{n} | {verdict} |")
        if flaky:
            lines += [
                "",
                "> 不稳定 case 是 harness 迭代最值得排查的对象："
                "同一任务多次运行结果不同。",
            ]

    lines += ["", "## 失败理由", ""]
    failures = [r for r in batch.results if r.status != "passed"]
    if not failures:
        lines.append("（无失败采样）")
    for r in failures:
        label = f"{r.case_id}（第 {r.attempt} 次）" if multi else r.case_id
        lines += [f"### {label} — {r.status}", ""]
        lines.append(f"理由：{r.reason}")
        if r.details:
            lines.append("")
            lines.extend(f"- {d}" for d in r.details)
        lines.append("")

    path = batch_dir / REPORT_FILE
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def load_trace_events(trace_path: Path) -> list[TraceEvent]:
    """读取一条轨迹 JSONL 并校验为标准事件。"""
    events: list[TraceEvent] = []
    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(trace_event_adapter.validate_python(json.loads(line)))
    return events


def _attempt_files(runs_dir: Path, case_id: str) -> list[tuple[int, Path]]:
    """收集一个 case 的全部采样轨迹，按采样序号升序。

    兼容两种历史布局：``<case>-a<N>.jsonl``（多次采样）与旧版
    ``<case>.jsonl``（单次，视为 attempt 1）。
    """
    found: list[tuple[int, Path]] = []
    legacy = runs_dir / f"{case_id}.jsonl"
    if legacy.exists():
        found.append((1, legacy))
    for path in runs_dir.glob(f"{case_id}-a*.jsonl"):
        try:
            attempt = int(path.stem.rsplit("-a", 1)[1])
        except ValueError:
            continue
        found.append((attempt, path))
    return sorted(found, key=lambda pair: pair[0])


def rebuild(batch_dir: Path) -> BatchResult:
    """对已存轨迹与产物重评分（不重新执行 Agent）。

    每 case 可能有多份采样轨迹（-a<attempt> 后缀），逐份重评，
    保持与运行时一致的 k/N 结构。
    """
    cases = load_cases(batch_dir)
    header: dict[str, Any] = {}
    results_path = batch_dir / RESULTS_FILE
    if results_path.exists():
        with open(results_path, encoding="utf-8") as fh:
            header = json.load(fh)

    results: list[CaseResult] = []
    runs_dir = batch_dir / "runs"
    for case in cases:
        attempts = _attempt_files(runs_dir, case.case_id) if runs_dir.is_dir() else []
        if not attempts:
            results.append(
                CaseResult(case_id=case.case_id, status="error", score=None,
                           reason="trace file missing")
            )
            continue

        for attempt, trace_path in attempts:
            events = load_trace_events(trace_path)
            finished = next(
                (ev for ev in reversed(events) if isinstance(ev, RunFinished)), None
            )
            if not events:
                results.append(
                    CaseResult(case_id=case.case_id, status="error", score=None,
                               reason="trace file empty", attempt=attempt)
                )
                continue
            if finished and finished.status in ("error", "timeout"):
                results.append(
                    CaseResult(case_id=case.case_id, status=finished.status,
                               score=None, attempt=attempt,
                               reason=f"execution terminated as {finished.status}")
                )
                continue
            snap = snapshot_from_events(events)
            scored = score_case(
                snap,
                check=case.check,
                expectations=case.expectations,
                work_dir=trace_path.parent.parent / "work" / trace_path.stem,
            )
            duration = (events[-1].ts - events[0].ts) if len(events) > 1 else 0.0
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    status=scored.status,
                    score=scored.score,
                    reason=scored.reason,
                    cost_usd=finished.cost_usd if finished else None,
                    turns=finished.turns if finished else 0,
                    duration_s=duration,
                    details=scored.details,
                    attempt=attempt,
                )
            )

    return BatchResult(
        batch_id=header.get("batch_id", batch_dir.name),
        dataset=header.get("dataset", batch_dir.name),
        model=header.get("model", "unknown"),
        started_at=header.get("started_at", ""),
        finished_at=header.get("finished_at", ""),
        results=results,
    )
