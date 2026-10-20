"""评测 CLI：拉数据、跑 Agent、出报告。

用法（在仓库根目录，经 uv workspace 运行）::

    uv run --package mcc-eval mcc-eval run --dataset humaneval --limit 5
    uv run --package mcc-eval mcc-eval run --cases my-cases.json --mock
    uv run --package mcc-eval mcc-eval report eval-runs/<批次目录>
    uv run --package mcc-eval mcc-eval datasets --dataset mbpp --limit 20 --out mbpp.json

模型解析复用 mcc 的约定：--model > MODEL 环境变量 > mcc settings.json
（model.main）> 平台默认值——评测不为模型配置引入第二份来源。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcc_eval.adapters import MccAdapter, MockAdapter
from mcc_eval.datasets import (
    DATASETS,
    Case,
    build_cases,
    export_cases_file,
    fetch_rows,
    load_cases_file,
)
from mcc_eval.report import rebuild, save_cases, write_report_md, write_results_json
from mcc_eval.runner import new_batch_dir, run_batch

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_OUT_ROOT = "eval-runs"
DEFAULT_CONCURRENCY = 2
DEFAULT_MAX_TURNS = 20
DEFAULT_TIMEOUT_S = 600.0


def resolve_model(explicit: str | None, *, settings_loader: Any | None = None) -> str:
    """按 mcc 的解析顺序确定模型，复用 CLI 已有配置。

    优先级与 mcc CLI 一致：显式参数 > MODEL 环境变量 > mcc settings
    （~/.mcc/settings.json 与 .mcc/settings.json 合并）的 model.main；
    mcc 配置读取失败时回落平台默认值。
    """
    if explicit:
        return explicit
    env = os.environ.get("MODEL")
    if env:
        return env
    if settings_loader is None:
        from mcc_config import load as load_settings

        try:
            settings = asyncio.run(load_settings(Path.cwd()))
        except Exception:
            return DEFAULT_MODEL
        try:
            return str(settings.model.main)
        except Exception:
            return DEFAULT_MODEL
    try:
        return str(settings_loader().model.main)
    except Exception:
        return DEFAULT_MODEL


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


def cmd_run(args: argparse.Namespace) -> int:
    cases, dataset_label = _gather_cases(args)
    if args.limit:
        cases = cases[: args.limit]

    model = "mock" if args.mock else resolve_model(args.model)
    batch_dir = new_batch_dir(
        Path(args.out), dataset_label, args.limit, len(cases)
    )
    save_cases(cases, batch_dir)

    adapter = MockAdapter() if args.mock else MccAdapter()
    print(f"模型: {model}｜Case: {len(cases)}｜并发: {args.concurrency}", file=sys.stderr)
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
