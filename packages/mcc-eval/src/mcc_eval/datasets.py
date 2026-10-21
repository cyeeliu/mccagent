"""业界基准数据集接入：拉取公开基准并转换为平台 Case。

支持三个高使用率的编码基准（数据经 HuggingFace datasets-server
公开 API 拉取，无需认证与额外依赖）：

  - swebench-lite   SWE-bench Lite（300 例）：真实 GitHub issue 修复任务
  - humaneval       HumanEval（164 例）：函数级 Python 代码生成经典基准
  - mbpp            MBPP test split（500 例）：基础 Python 编程任务

HumanEval / MBPP 的转换会保留官方测试代码（``check`` 字段），评分走
执行式验证——Agent 产物 + 官方测试在子进程中运行，通过即 passed，
对齐官方 pass@1 的单样本语义。SWE-bench 的官方评测需要 Docker 任务
容器，本平台退化为行为断言评分（tool_called / no_tool_errors /
output_contains），语义是「Agent 是否按指令完成了 clone / 修复 /
自检 / 汇报」，不等价于官方解决率——报告会显式标注。

网络拉取带指数退避重试（datasets-server 偶发 502 / 连接重置）；
重试耗尽抛 RuntimeError，由 CLI 转成带离线指引的错误退出。

离线场景（受限网络）：先在联网机器导出 ``--out cases.json``，再在
目标机器用 ``run --cases cases.json`` 运行。
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
FETCH_ATTEMPTS = 3  # 单页请求总尝试次数（1 次首发 + 2 次重试）
RETRY_BASE_DELAY_S = 1.0

# 瞬时故障：连接重置 / 超时 / 协议错误，重试大概率成功
RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)

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


@dataclass
class Case:
    """一个待评测任务：提示词 + 环境预置 + 可选的官方测试与行为断言。

    check 存在时评分走执行式（官方测试子进程运行，二值判定）；
    否则走 expectations 行为断言（加权得分）。两者可以并存——
    执行式决定 status/score，断言明细进 details 供诊断。

    setup 是声明式的环境预置：``{相对路径: 文件内容}``
    （路径限定在工作目录内），Agent 开跑前逐个写入。
    评测任务因此可以表达「给定环境快照，改出正确终态」的
    回流 case，而不只是裸任务描述。
    """

    case_id: str
    task: str
    check: str | None = None
    expectations: list[dict[str, Any]] = field(default_factory=list)
    setup: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "check": self.check,
            "expectations": self.expectations,
            "setup": self.setup,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Case:
        case_id = str(raw.get("case_id") or "").strip()
        task = str(raw.get("task") or "")
        if not case_id or not task.strip():
            raise ValueError("case needs case_id and task")
        setup_raw = raw.get("setup") or {}
        if not isinstance(setup_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in setup_raw.items()
        ):
            raise ValueError(
                "case setup must be a {relative_path: content} mapping of strings"
            )
        return cls(
            case_id=case_id,
            task=task,
            check=raw.get("check") or None,
            expectations=list(raw.get("expectations") or []),
            setup=dict(setup_raw),
        )


@dataclass(frozen=True)
class DatasetSpec:
    """一个公开基准的拉取与转换规格。"""

    hf_id: str
    config: str
    split: str
    title: str
    converter: Callable[[dict[str, Any]], Case | None]


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


def humaneval_case(row: dict[str, Any]) -> Case | None:
    """HumanEval 记录 → Case（写实现文件 + 自检 + 汇报 + 官方测试执行评分）。"""
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
    check = (
        f"from solution import {entry_point} as candidate\n\n"
        f"{test}\n\n"
        "check(candidate)\n"
        "print('PASSED')\n"
    )
    return Case(
        case_id=str(task_id).replace("/", "_"),
        task=task,
        check=check,
        expectations=[dict(e) for e in CODE_GEN_EXPECTATIONS],
    )


def mbpp_case(row: dict[str, Any]) -> Case | None:
    """MBPP 记录 → Case（写实现文件 + 自检 + 汇报 + 官方断言执行评分）。"""
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
        f"1. 用 {TOOL_WRITE} 工具把实现写入 solution.py（顶层函数名须与断言一致）。\n"
        f"2. 用 {TOOL_BASH} 运行下方全部断言验证实现"
        "（可先写入 check.py 再执行 python check.py）。\n"
        f"3. 全部断言通过后，在最终回复中单独输出一行：PASSED\n\n"
        f"测试代码：\n{tests}"
    )
    check = f"from solution import *\n\n{tests}\nprint('PASSED')\n"
    return Case(
        case_id=f"mbpp_{task_id}",
        task=task,
        check=check,
        expectations=[dict(e) for e in CODE_GEN_EXPECTATIONS],
    )


def swebench_case(row: dict[str, Any]) -> Case | None:
    """SWE-bench Lite 记录 → Case（clone 仓库 + 修复 + 测试 + 汇报）。

    官方评测需 Docker 任务容器，此处用行为断言做代理评分。
    """
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
    return Case(
        case_id=str(instance_id),
        task=task,
        check=None,
        expectations=[dict(e) for e in BUGFIX_EXPECTATIONS],
    )


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


def _retryable(exc: Exception) -> bool:
    """是否值得重试：瞬时传输错误，或 429 / 5xx 状态码。"""
    if isinstance(exc, RETRYABLE_TRANSPORT_ERRORS):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _get_with_retry(
    client: httpx.Client, params: dict[str, Any], *, what: str
) -> httpx.Response:
    """单次 rows API 请求：指数退避重试瞬时故障。

    重试耗尽把原始异常包进 RuntimeError（附已尝试的请求与次数），
    由调用方决定如何呈现——库层不直接退出进程。
    """
    last_exc: Exception | None = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            resp = client.get(ROWS_API, params=params)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if not _retryable(exc):
                raise
            last_exc = exc
            if attempt < FETCH_ATTEMPTS:
                delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                print(
                    f"拉取 {what} 第 {attempt} 次失败"
                    f"（{type(exc).__name__}: {exc}），"
                    f"{delay:.0f}s 后重试…",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"拉取 {what} 失败（已重试 {FETCH_ATTEMPTS} 次）：{last_exc}"
    ) from last_exc


def fetch_rows(spec: DatasetSpec, limit: int = 0, client: httpx.Client | None = None) -> list[dict]:
    """分页拉取数据集原始记录；limit=0 表示全量。

    每页请求对瞬时网络故障（连接重置 / 超时）与 429 / 5xx 做指数
    退避重试；重试耗尽抛 RuntimeError——跨国访问 datasets-server
    连接被重置是常态，不该以裸 traceback 收场。
    """
    rows: list[dict] = []
    own = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        offset = 0
        while True:
            resp = _get_with_retry(
                client,
                {
                    "dataset": spec.hf_id,
                    "config": spec.config,
                    "split": spec.split,
                    "offset": offset,
                    "length": PAGE_SIZE,
                },
                what=f"{spec.hf_id}（offset={offset}）",
            )
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


def build_cases(dataset: str, records: list[dict]) -> list[Case]:
    """原始记录列表 → Case 列表（跳过字段不完整的记录）。"""
    spec = DATASETS[dataset]
    cases = [case for record in records if (case := spec.converter(record)) is not None]
    if not cases:
        raise ValueError(f"dataset {dataset!r} yielded no valid cases")
    return cases


def load_cases_file(path: str | Path) -> list[Case]:
    """从本地 JSON 文件加载 Case（离线运行 / 自定义任务）。"""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of cases")
    return [Case.from_dict(item) for item in raw]


def export_cases_file(cases: list[Case], path: str | Path) -> None:
    """导出 Case 列表为 JSON（供离线机器 ``run --cases`` 使用）。"""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in cases], fh, ensure_ascii=False, indent=2)
