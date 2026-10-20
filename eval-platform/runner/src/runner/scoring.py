"""评分器：根据 expectations 对轨迹快照打分。

评分策略按权重叠加，支持四类断言：
  1. ``tool_called``      工具调用断言（工具名 / 参数正则 / 顺序）
  2. ``output_contains``  最终输出文本断言（大小写不敏感 substring / regex）
  3. ``no_tool_errors``   执行无错误奖励
  4. ``tool_call_count``  工具调用次数区间断言

期望格式（batches.cases[i].expectations）::

    [
      {"type": "tool_called", "tool": "bash",
       "args_regex": "pytest", "weight": 0.4},
      {"type": "output_contains", "pattern": "passed", "weight": 0.4},
      {"type": "no_tool_errors", "weight": 0.2}
    ]

未配置 expectations 时视为固定通过，score 为 None（留给人工 review）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from runner.events import ToolCall, ToolResult

DEFAULT_PASS_THRESHOLD = 0.6


@dataclass
class TraceSnapshot:
    """从事件流中提取的 run 摘要，传给 score()。"""

    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    text_outputs: list[str] = field(default_factory=list)  # 所有 LlmChunk.delta 的拼接
    had_error: bool = False  # 任意 ToolResult.is_error


@dataclass
class ScoreResult:
    status: str  # "passed" | "failed"
    score: float | None  # None 表示没有 expectations
    reason: str


def score(snapshot: TraceSnapshot, expectations: list[dict[str, Any]]) -> ScoreResult:
    """评分入口。"""
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
    return ScoreResult(status=status, score=round(ratio, 4), reason="; ".join(reasons) or "ok")


def _check_tool_called(snap: TraceSnapshot, exp: dict[str, Any]) -> tuple[bool, str]:
    """验证某工具是否被调用（可选匹配参数正则、可选顺序约束）。"""
    tool_name = exp.get("tool", "")
    args_regex = exp.get("args_regex")
    ordered_after = exp.get("after_tool")  # 要求在 after_tool 之后调用

    matched_idx = [
        i for i, tc in enumerate(snap.tool_calls) if tc.tool_name == tool_name
    ]
    if not matched_idx:
        return False, f"tool '{tool_name}' not called"

    if args_regex:
        pattern = re.compile(args_regex, re.IGNORECASE)
        matched_idx = [
            i
            for i in matched_idx
            if pattern.search(str(snap.tool_calls[i].args or ""))
        ]
        if not matched_idx:
            return False, f"tool '{tool_name}' called but args did not match /{args_regex}/"

    if ordered_after:
        after_idxs = [
            i for i, tc in enumerate(snap.tool_calls) if tc.tool_name == ordered_after
        ]
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
