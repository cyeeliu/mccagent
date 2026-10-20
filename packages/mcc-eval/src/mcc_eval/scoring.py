"""评分器：执行式优先，行为断言兜底。

两种评分策略：

  1. 执行式（case.check 存在，HumanEval / MBPP）：把官方测试代码写入
     Agent 工作目录，用独立子进程运行（超时保护），退出码 0 即 passed。
     语义对齐官方基准的单样本判定——Agent 产物 + 官方测试，测试说了算。
  2. 断言式（case.expectations）：对轨迹快照做加权断言
     （tool_called / output_contains / no_tool_errors / tool_call_count）。

两种可以并存：执行式决定最终 status / score，断言明细进 details
供失败诊断。执行式判定不依赖 Agent 自述（如「PASSED」文本），从产物
层面杜绝「声称通过但代码错误」的假阳性。

子进程运行不是安全沙箱（与 HumanEval 官方 harness 的取舍一致）：
仅运行可信来源的测试代码，勿对不可信 case 开放。
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcc_eval.events import ToolCall, ToolResult

DEFAULT_PASS_THRESHOLD = 0.6
CHECK_TIMEOUT_S = 30.0
CHECK_FILE_NAME = "_eval_check.py"
OUTPUT_TAIL_CHARS = 800
# -I 隔离模式同时移除脚本目录的 sys.path 注入，头部显式恢复对
# 工作目录（Agent 产物 solution.py 所在处）的导入能力
CHECK_PREAMBLE = (
    "import sys as _sys, pathlib as _pathlib\n"
    "_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))\n"
)


@dataclass
class TraceSnapshot:
    """从事件流中提取的 run 摘要，传给断言式评分。"""

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    text_outputs: list[str] = field(default_factory=list)  # 所有 LlmChunk.delta 的拼接
    had_error: bool = False  # 任意 ToolResult.is_error


@dataclass
class ScoreResult:
    """一次评分的完整结论。"""

    status: str  # "passed" | "failed"
    score: float | None  # None 表示没有可评分内容
    reason: str
    details: list[str] = field(default_factory=list)  # 断言明细（诊断用）


def score_case(
    snapshot: TraceSnapshot,
    *,
    check: str | None,
    expectations: list[dict[str, Any]],
    work_dir: Path,
    timeout_s: float = CHECK_TIMEOUT_S,
) -> ScoreResult:
    """评分入口：check 存在走执行式，否则走断言式。"""
    assertion = score_assertions(snapshot, expectations)
    if check is None:
        return assertion

    ok, output = run_execution_check(work_dir, check, timeout_s=timeout_s)
    status = "passed" if ok else "failed"
    reason = (
        f"execution check {'passed' if ok else 'failed'}; "
        f"exit output tail: {output or '(empty)'}"
    )
    return ScoreResult(status=status, score=1.0 if ok else 0.0, reason=reason,
                       details=assertion.details)


def run_execution_check(
    work_dir: Path,
    check_code: str,
    *,
    timeout_s: float = CHECK_TIMEOUT_S,
    check_file_name: str = CHECK_FILE_NAME,
) -> tuple[bool, str]:
    """在工作目录运行官方测试代码，返回 (是否通过, 输出尾部)。

    -I（isolated mode）隔离 PYTHONPATH 等环境注入；工作目录经头部
    preamble 显式注册回 sys.path，Agent 产物 solution.py 由此加载。
    """
    test_file = work_dir / check_file_name
    test_file.write_text(CHECK_PREAMBLE + check_code, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", test_file.name],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s}s"
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output[-OUTPUT_TAIL_CHARS:]


def score_assertions(
    snapshot: TraceSnapshot, expectations: list[dict[str, Any]]
) -> ScoreResult:
    """断言式评分：按权重叠加。"""
    if not expectations:
        return ScoreResult(status="passed", score=None, reason="no expectations; manual review")

    total_weight = 0.0
    earned = 0.0
    reasons: list[str] = []

    for exp in expectations:
        etype = exp.get("type", "")
        weight = float(exp.get("weight", 1.0))
        total_weight += weight

        if etype == "tool_called":
            ok, why = _check_tool_called(snapshot, exp)
        elif etype == "output_contains":
            ok, why = _check_output_contains(snapshot, exp)
        elif etype == "no_tool_errors":
            ok, why = (
                not snapshot.had_error,
                ("no tool errors" if not snapshot.had_error else "had tool errors"),
            )
        elif etype == "tool_call_count":
            ok, why = _check_tool_count(snapshot, exp)
        else:
            # 未知 expectation 类型：宽松跳过，不影响得分
            total_weight -= weight
            continue

        if ok:
            earned += weight
            reasons.append(f"✓ {etype}: {why}")
        else:
            reasons.append(f"✗ {etype}: {why}")

    if total_weight == 0:
        return ScoreResult(status="passed", score=None, reason="no scorable expectations")

    ratio = earned / total_weight
    threshold = float(expectations[0].get("pass_threshold", DEFAULT_PASS_THRESHOLD))
    status = "passed" if ratio >= threshold else "failed"
    return ScoreResult(status=status, score=round(ratio, 4), reason="; ".join(reasons) or "ok",
                       details=reasons)


def _check_tool_called(snap: TraceSnapshot, exp: dict[str, Any]) -> tuple[bool, str]:
    """验证某工具是否被调用（可选匹配参数正则、可选顺序约束）。"""
    tool_name = exp.get("tool", "")
    args_regex = exp.get("args_regex")
    ordered_after = exp.get("after_tool")  # 要求在 after_tool 之后调用

    matched_idx = [i for i, tc in enumerate(snap.tool_calls) if tc.tool_name == tool_name]
    if not matched_idx:
        return False, f"tool '{tool_name}' not called"

    if args_regex:
        pattern = re.compile(args_regex, re.IGNORECASE)
        matched_idx = [
            i for i in matched_idx if pattern.search(str(snap.tool_calls[i].args or ""))
        ]
        if not matched_idx:
            return False, f"tool '{tool_name}' called but args did not match /{args_regex}/"

    if ordered_after:
        after_idxs = [i for i, tc in enumerate(snap.tool_calls) if tc.tool_name == ordered_after]
        # 任意一次匹配调用出现在 after_tool 的某次调用之后即满足——工具
        # 被多次调用时，只看第一次会把「后置调用」误判为顺序不符
        if not any(i > j for i in matched_idx for j in after_idxs):
            return False, f"tool '{tool_name}' not called after '{ordered_after}'"

    return True, f"tool '{tool_name}' called {len(matched_idx)} time(s)"


def _check_output_contains(snap: TraceSnapshot, exp: dict[str, Any]) -> tuple[bool, str]:
    """验证文本输出包含某 substring 或匹配某正则。"""
    pattern_str = exp.get("pattern", "")
    use_regex = exp.get("regex", False)
    full_text = " ".join(snap.text_outputs)

    if use_regex:
        ok = bool(re.search(pattern_str, full_text, re.IGNORECASE))
        return ok, f"regex /{pattern_str}/ {'found' if ok else 'not found'}"
    ok = pattern_str.lower() in full_text.lower()
    return ok, f"'{pattern_str}' {'found' if ok else 'not found'} in output"


def _check_tool_count(snap: TraceSnapshot, exp: dict[str, Any]) -> tuple[bool, str]:
    """验证某工具调用次数在 [min, max] 范围内。"""
    tool_name = exp.get("tool", "")
    min_c = exp.get("min", 0)
    max_c = exp.get("max", 999_999)
    count = sum(1 for tc in snap.tool_calls if tc.tool_name == tool_name)
    ok = min_c <= count <= max_c
    return ok, f"tool '{tool_name}' called {count} time(s) (expected [{min_c}, {max_c}])"
