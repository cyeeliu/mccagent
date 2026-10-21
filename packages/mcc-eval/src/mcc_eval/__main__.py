"""评测 CLI：拉数据、跑 Agent、出报告、回流失败会话。

用法（在仓库根目录，经 uv workspace 运行）::

    uv run --package mcc-eval mcc-eval run --dataset humaneval --limit 5
    uv run --package mcc-eval mcc-eval run --cases my-cases.json --mock
    uv run --package mcc-eval mcc-eval run --cases my-cases.json --runs-per-case 3
    uv run --package mcc-eval mcc-eval report eval-runs/<批次目录>
    uv run --package mcc-eval mcc-eval datasets --dataset mbpp --limit 20 --out mbpp.json
    uv run --package mcc-eval mcc-eval distill --limit 20

模型解析与 provider 构建统一走 mcc-config（resolve_model /
build_llm_provider），与 mcc CLI 同一入口同一优先级——评测不为
模型配置引入第二份来源。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mcc_config import resolve_model

from mcc_eval.adapters import MccAdapter, MockAdapter
from mcc_eval.datasets import (
    DATASETS,
    Case,
    build_cases,
    export_cases_file,
    fetch_rows,
    load_cases_file,
)
from mcc_eval.distill import distill_sessions
from mcc_eval.report import rebuild, save_cases, write_report_md, write_results_json
from mcc_eval.runner import new_batch_dir, run_batch

DEFAULT_OUT_ROOT = "eval-runs"
DEFAULT_CONCURRENCY = 2
# 默认轮数对齐 mcc BudgetConfig.max_iterations（40）：20 轮连
# 「clone 仓库 → 定位 → 修复 → 跑测试」的完整 SWE-bench 流程都走不完
DEFAULT_MAX_TURNS = 40
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_RUNS_PER_CASE = 1


def _non_negative_int(raw: str) -> int:
    """argparse 类型：非负整数（--limit 传负数会静默截断尾部记录）。"""
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _gather_cases(args: argparse.Namespace) -> tuple[list[Case], str]:
    """按参数取 Case 来源：--cases 本地文件或 --dataset 在线拉取。"""
    if args.cases:
        return load_cases_file(args.cases), Path(args.cases).stem
    if args.dataset not in DATASETS:
        raise SystemExit(f"未知数据集 {args.dataset!r}，可选：{', '.join(sorted(DATASETS))}")
    spec = DATASETS[args.dataset]
    print(f"拉取 {spec.title}（{spec.hf_id}）…", file=sys.stderr)
    records = fetch_rows(spec, args.limit)
    print(f"取得 {len(records)} 条记录", file=sys.stderr)
    return build_cases(args.dataset, records), args.dataset


def _positive_int(raw: str) -> int:
    """argparse 类型：正整数（采样次数至少 1 才构成一次结论）。"""
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def cmd_run(args: argparse.Namespace) -> int:
    cases, dataset_label = _gather_cases(args)
    if args.limit:
        cases = cases[: args.limit]

    model = "mock" if args.mock else resolve_model("main", args.model)
    batch_dir = new_batch_dir(
        Path(args.out), dataset_label, args.limit, len(cases)
    )
    save_cases(cases, batch_dir)

    adapter = MockAdapter() if args.mock else MccAdapter()
    print(
        f"模型: {model}｜Case: {len(cases)}｜并发: {args.concurrency}"
        f"｜采样: {args.runs_per_case}/case",
        file=sys.stderr,
    )
    print(f"批次目录: {batch_dir}", file=sys.stderr)

    batch = asyncio.run(
        run_batch(
            cases,
            adapter,
            dataset=dataset_label,
            model=model,
            batch_dir=batch_dir,
            concurrency=args.concurrency,
            max_turns=args.max_turns,
            timeout_s=args.timeout,
            runs_per_case=args.runs_per_case,
        )
    )
    write_results_json(batch, batch_dir)
    spec = DATASETS.get(dataset_label)
    write_report_md(batch, batch_dir, title=spec.title if spec else dataset_label)
    print(
        f"完成：passed {batch.passed}/{batch.total}，通过率 {batch.pass_rate:.1%}；"
        f"报告 {batch_dir}/report.md",
        file=sys.stderr,
    )
    if batch.flaky_cases:
        print(
            f"注意：{batch.flaky_cases} 个 case 采样结果不稳定（flaky），"
            f"详见报告的 case 级聚合表",
            file=sys.stderr,
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        raise SystemExit(f"批次目录不存在：{batch_dir}")
    batch = rebuild(batch_dir)
    write_results_json(batch, batch_dir)
    spec = DATASETS.get(batch.dataset)
    write_report_md(batch, batch_dir, title=spec.title if spec else batch.dataset)
    print(
        f"重评完成：passed {batch.passed}/{batch.total}，通过率 {batch.pass_rate:.1%}；"
        f"报告 {batch_dir}/report.md",
        file=sys.stderr,
    )
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    if args.out:
        if args.dataset not in DATASETS:
            raise SystemExit("--out 需要配合 --dataset 使用")
        spec = DATASETS[args.dataset]
        records = fetch_rows(spec, args.limit)
        cases = build_cases(args.dataset, records)
        export_cases_file(cases, args.out)
        print(f"已导出 {len(cases)} 个 case 到 {args.out}", file=sys.stderr)
        return 0
    for key, spec in sorted(DATASETS.items()):
        print(f"{key:14} {spec.title}（{spec.hf_id}，{spec.split} split）")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """失败回流：把生产会话固化成 case 草稿。"""
    sessions_dir = Path(args.sessions_dir)
    if not sessions_dir.is_dir():
        raise SystemExit(f"会话目录不存在：{sessions_dir}")
    statuses = tuple(s.strip() for s in args.status.split(",") if s.strip())
    outcomes = distill_sessions(
        sessions_dir,
        statuses=statuses,
        limit=args.limit,
        out_dir=Path(args.out),
    )
    created = [o for o in outcomes if o.created]
    skipped = [o for o in outcomes if not o.created]
    print(
        f"回流完成：新建 {len(created)}，跳过（已存在）{len(skipped)}，"
        f"状态过滤 {statuses}；草稿目录 {args.out}",
        file=sys.stderr,
    )
    for o in created:
        print(f"  + {o.path}（源状态 {o.source_status}）", file=sys.stderr)
    for o in skipped:
        print(f"  = {o.path}（已存在，未覆盖）", file=sys.stderr)
    if created:
        print(
            "草稿的 expectations / setup 为空，需人工补齐判定器并校准后"
            "再入正式评测集",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcc-eval", description="轻量评测：业界基准 + mcc Agent 轨迹与评分"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行一批评测 case")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=sorted(DATASETS), help="业界基准数据集")
    source.add_argument("--cases", help="本地 case JSON 文件（离线 / 自定义任务）")
    run.add_argument("--limit", type=_non_negative_int, default=0,
                     help="只取前 N 个 case（0 = 全量）")
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                     help=f"并发数（默认 {DEFAULT_CONCURRENCY}）")
    run.add_argument("--runs-per-case", type=_positive_int,
                     default=DEFAULT_RUNS_PER_CASE,
                     help=f"每 case 采样次数（默认 {DEFAULT_RUNS_PER_CASE}；"
                          f"Agent 输出非确定，≥3 才能看出稳定性）")
    run.add_argument("--model", default=None,
                     help="覆盖模型（缺省复用 mcc 配置：MODEL 环境变量 > settings.json）")
    run.add_argument("--mock", action="store_true",
                     help="用 Mock 适配器冒烟链路（无需 API key）")
    run.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                     help=f"单 case 最大 Agent 轮数（默认 {DEFAULT_MAX_TURNS}）")
    run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                     help=f"单 case 超时秒数（默认 {DEFAULT_TIMEOUT_S}）")
    run.add_argument("--out", default=DEFAULT_OUT_ROOT,
                     help=f"产物根目录（默认 {DEFAULT_OUT_ROOT}/）")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="对已存轨迹与产物重评分")
    report.add_argument("batch_dir", help="批次目录（含 cases.json 与 runs/）")
    report.set_defaults(func=cmd_report)

    datasets = sub.add_parser("datasets", help="列出 / 导出支持的基准数据集")
    datasets.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    datasets.add_argument("--limit", type=_non_negative_int, default=0)
    datasets.add_argument("--out", default=None, help="导出 case JSON（离线运行用）")
    datasets.set_defaults(func=cmd_datasets)

    distill = sub.add_parser(
        "distill", help="把失败生产会话回流为评测 case 草稿"
    )
    distill.add_argument(
        "--sessions-dir", default=str(Path.home() / ".mcc" / "sessions"),
        help="mcc 会话目录（默认 ~/.mcc/sessions）",
    )
    distill.add_argument(
        "--status", default="errored",
        help="按会话状态过滤，逗号分隔（默认 errored）",
    )
    distill.add_argument("--limit", type=_non_negative_int, default=20,
                         help="最多处理多少个会话（默认 20）")
    distill.add_argument("--out", default="evals/cases",
                         help="草稿落盘目录（默认 evals/cases/）")
    distill.set_defaults(func=cmd_distill)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
