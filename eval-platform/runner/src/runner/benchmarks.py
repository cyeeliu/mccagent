"""业界评测集接入：拉取公开基准数据并转换为平台批次。

支持三个高使用率的编码 Agent 基准（数据经 HuggingFace datasets-server
公开 API 拉取，无需认证与额外依赖）：

  - swebench-lite   SWE-bench Lite（300 例）：真实 GitHub issue 修复任务，
                    编码 Agent 领域的事实标准
  - humaneval       HumanEval（164 例）：函数级 Python 代码生成经典基准
  - mbpp            MBPP test split（500 例）：基础 Python 编程任务

转换后的 case 按行为断言评分（tool_called / no_tool_errors /
output_contains），语义是「Agent 是否按指令完成了写码 / 自检 / 汇报」，
不等价于官方 pass@1 —— 后者需要独立的测试容器做执行式验证。

用法（runner 容器内，compose 启动之后）::

    python -m runner.benchmarks --dataset swebench-lite --limit 10 --submit

本地导出批次 JSON（不提交）::

    python -m runner.benchmarks --dataset humaneval --out batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

DEFAULT_PROFILE = "mcc-main"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_PARALLELISM = 2

# mcc 内建工具名（与被测 Agent 的工具注册表一致）
TOOL_WRITE = "write_file"
TOOL_BASH = "run_bash"

CODE_GEN_EXPECTATIONS = [
    {"type": "tool_called", "tool": TOOL_WRITE, "weight": 0.2},
    {"type": "tool_called", "tool": TOOL_BASH, "args_regex": "python", "weight": 0.3},
    {"type": "no_tool_errors", "weight": 0.2},
    {"type": "output_contains", "pattern": "PASSED", "weight": 0.3},
]

BUGFIX_EXPECTATIONS = [
    {"type": "tool_called", "tool": TOOL_BASH, "args_regex": "git clone", "weight": 0.3},
    {"type": "no_tool_errors", "weight": 0.3},
    {"type": "output_contains", "pattern": "DONE", "weight": 0.4},
]


@dataclass(frozen=True)
class DatasetSpec:
    """一个公开基准的拉取与转换规格。"""

    hf_id: str
    config: str
    split: str
    title: str
    converter: Callable[[dict[str, Any]], dict[str, Any] | None]


def _as_list(value: Any) -> list[str]:
    """HF 数据集里列表字段可能被序列化为 JSON 字符串，两种形态都接受。"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def humaneval_case(row: dict[str, Any]) -> dict[str, Any] | None:
    """HumanEval 记录 → 平台 case（写实现文件 + 自检 + 汇报）。"""
    task_id = row.get("task_id")
    prompt = row.get("prompt")
    test = row.get("test")
    entry_point = row.get("entry_point")
    if not all(isinstance(v, str) and v.strip() for v in (task_id, prompt, test, entry_point)):
        return None
    task = (
        f"实现函数 {entry_point}，使其通过下方全部单元测试（HumanEval {task_id}）。\n\n"
        f"函数签名与说明：\n{prompt}\n\n"
        f"要求：\n"
        f"1. 用 {TOOL_WRITE} 工具把完整实现写入 solution.py（含必要的 import）。\n"
        f"2. 用 {TOOL_BASH} 运行下方测试代码验证实现"
        "（可先写入 check.py 再执行 python check.py）。\n"
        f"3. 全部断言通过后，在最终回复中单独输出一行：PASSED\n\n"
        f"测试代码：\n{test}"
    )
    return {
        "case_id": str(task_id).replace("/", "_"),
        "task": task,
        "expectations": [dict(e) for e in CODE_GEN_EXPECTATIONS],
    }


def mbpp_case(row: dict[str, Any]) -> dict[str, Any] | None:
    """MBPP 记录 → 平台 case（写实现文件 + 自检 + 汇报）。"""
    task_id = row.get("task_id")
    text = row.get("text")
    test_list = _as_list(row.get("test_list"))
    if not isinstance(task_id, (int, str)) or not isinstance(text, str) or not text.strip():
        return None
    if not test_list:
        return None
    tests = "\n".join(test_list)
    task = (
        f"用 Python 完成以下任务（MBPP #{task_id}）：\n\n"
        f"{text}\n\n"
        f"要求：\n"
        f"1. 用 {TOOL_WRITE} 工具把实现写入 solution.py。\n"
        f"2. 用 {TOOL_BASH} 运行下方全部断言验证实现"
        "（可先写入 check.py 再执行 python check.py）。\n"
        f"3. 全部断言通过后，在最终回复中单独输出一行：PASSED\n\n"
        f"测试代码：\n{tests}"
    )
    return {
        "case_id": f"mbpp_{task_id}",
        "task": task,
        "expectations": [dict(e) for e in CODE_GEN_EXPECTATIONS],
    }


def swebench_case(row: dict[str, Any]) -> dict[str, Any] | None:
    """SWE-bench Lite 记录 → 平台 case（clone 仓库 + 修复 + 测试 + 汇报）。"""
    instance_id = row.get("instance_id")
    repo = row.get("repo")
    problem = row.get("problem_statement")
    if not all(isinstance(v, str) and v.strip() for v in (instance_id, repo, problem)):
        return None
    fail_to_pass = _as_list(row.get("FAIL_TO_PASS"))
    tests = "\n".join(f"- {t}" for t in fail_to_pass) or "- （数据集未提供，请自行定位相关测试）"
    task = (
        f"修复仓库 {repo}（SWE-bench Lite {instance_id}）中的以下问题：\n\n"
        f"{problem}\n\n"
        f"要求：\n"
        f"1. 用 {TOOL_BASH} 执行 git clone https://github.com/{repo}.git repo 获取工作副本。\n"
        f"2. 阅读相关代码，定位并修复上述问题。\n"
        f"3. 运行下方测试确认修复生效（进入 repo 目录后用 python -m pytest 执行）。\n"
        f"4. 完成后在最终回复中单独输出一行：DONE\n\n"
        f"需要通过的测试：\n{tests}"
    )
    return {
        "case_id": str(instance_id),
        "task": task,
        "expectations": [dict(e) for e in BUGFIX_EXPECTATIONS],
    }


DATASETS: dict[str, DatasetSpec] = {
    "swebench-lite": DatasetSpec(
        hf_id="princeton-nlp/SWE-bench_Lite",
        config="default",
        split="test",
        title="SWE-bench Lite",
        converter=swebench_case,
    ),
    "humaneval": DatasetSpec(
        hf_id="openai/openai_humaneval",
        config="default",
        split="test",
        title="HumanEval",
        converter=humaneval_case,
    ),
    "mbpp": DatasetSpec(
        hf_id="google-research-datasets/mbpp",
        config="default",
        split="test",
        title="MBPP",
        converter=mbpp_case,
    ),
}


def fetch_rows(spec: DatasetSpec, limit: int = 0, client: httpx.Client | None = None) -> list[dict]:
    """分页拉取数据集原始记录；limit=0 表示全量。"""
    rows: list[dict] = []
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        offset = 0
        while True:
            resp = client.get(
                ROWS_API,
                params={
                    "dataset": spec.hf_id,
                    "config": spec.config,
                    "split": spec.split,
                    "offset": offset,
                    "length": PAGE_SIZE,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            page = [item["row"] for item in payload.get("rows", [])]
            rows.extend(page)
            if not page:
                break
            if limit and len(rows) >= limit:
                break
            if len(rows) >= payload.get("num_rows_total", 0):
                break
            offset += len(page)
    finally:
        if own:
            client.close()
    return rows[:limit] if limit else rows


def build_batch(
    dataset: str,
    records: list[dict],
    *,
    name: str | None = None,
    profile: str = DEFAULT_PROFILE,
    parallelism: int = DEFAULT_PARALLELISM,
) -> dict[str, Any]:
    """原始记录列表 → 平台批次请求体（跳过字段不完整的记录）。"""
    spec = DATASETS[dataset]
    cases = []
    skipped = 0
    for record in records:
        case = spec.converter(record)
        if case is None:
            skipped += 1
            continue
        cases.append(case)
    if not cases:
        raise ValueError(f"dataset {dataset!r} yielded no valid cases")
    if skipped:
        print(f"warning: skipped {skipped} incomplete record(s)", file=sys.stderr)
    return {
        "name": name or f"{dataset}-{len(cases)}",
        "profile": profile,
        "parallelism": parallelism,
        "cases": cases,
    }


def submit_batch(
    batch: dict[str, Any],
    *,
    server: str,
    model: str = DEFAULT_MODEL,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """幂等创建 profile 并提交批次，返回批次响应。"""
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        server = server.rstrip("/")
        profiles = client.get(f"{server}/api/profiles")
        profiles.raise_for_status()
        existing = {p["name"] for p in profiles.json().get("items", [])}
        if batch["profile"] not in existing:
            resp = client.post(
                f"{server}/api/profiles",
                json={
                    "name": batch["profile"],
                    "scaffold": "mini-claude-code",
                    "model": model,
                    "harness_version": "dev",
                    "sandbox_image": "local",
                },
            )
            resp.raise_for_status()
        resp = client.post(f"{server}/api/batches", json=batch)
        resp.raise_for_status()
        return resp.json()
    finally:
        if own:
            client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m runner.benchmarks",
        description="拉取业界公开编码基准并转换为评测批次",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="基准数据集",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="截取前 N 条记录（0 = 全量，默认 10）",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("EVAL_SERVER_URL", "http://localhost:8081"),
        help="API 服务地址（默认读 EVAL_SERVER_URL）",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Agent profile 名称")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="profile 使用的模型")
    parser.add_argument("--parallelism", type=int, default=DEFAULT_PARALLELISM)
    parser.add_argument("--name", default=None, help="批次名称（默认 <dataset>-<count>）")
    parser.add_argument("--out", default=None, help="批次 JSON 导出路径（不提交）")
    parser.add_argument("--submit", action="store_true", help="提交到 API 服务")
    parser.add_argument(
        "--web-url",
        default=os.environ.get("EVAL_WEB_URL", "http://localhost:5173"),
        help="web 看板地址（用于提示批次页面链接）",
    )
    args = parser.parse_args(argv)

    if not args.submit and not args.out:
        parser.error("需要 --submit 或 --out 至少其一")

    spec = DATASETS[args.dataset]
    print(f"拉取 {spec.title}（{spec.hf_id}）…", file=sys.stderr)
    records = fetch_rows(spec, args.limit)
    print(f"取得 {len(records)} 条记录", file=sys.stderr)
    batch = build_batch(
        args.dataset,
        records,
        name=args.name,
        profile=args.profile,
        parallelism=args.parallelism,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(batch, fh, ensure_ascii=False, indent=2)
        print(f"批次已导出: {args.out}（{len(batch['cases'])} 例）", file=sys.stderr)

    if args.submit:
        result = submit_batch(batch, server=args.server, model=args.model)
        batch_id = result.get("id")
        print(f"批次已提交: {batch_id}")
        print(f"看板页面: {args.web_url}/batches/{batch_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
