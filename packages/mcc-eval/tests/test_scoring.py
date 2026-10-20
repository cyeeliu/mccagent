"""评分器测试：断言式全家桶与执行式（真实子进程）。"""

from __future__ import annotations

import pytest
from mcc_eval.events import ToolCall, ToolResult
from mcc_eval.scoring import (
    TraceSnapshot,
    run_execution_check,
    score_assertions,
    score_case,
)


def _snap(**overrides) -> TraceSnapshot:
    snap = TraceSnapshot(
        tool_calls=[
            ToolCall(seq=0, turn=0, call_id="a", tool_name="bash", args="python -m pytest"),
            ToolCall(seq=1, turn=0, call_id="b", tool_name="edit", args={}),
            ToolCall(seq=2, turn=1, call_id="c", tool_name="bash", args="python check.py"),
        ],
        tool_results=[ToolResult(seq=3, call_id="a", output="ok")],
        text_outputs=["task done", "PASSED"],
    )
    for key, value in overrides.items():
        setattr(snap, key, value)
    if "had_error" not in overrides:
        snap.had_error = any(r.is_error for r in snap.tool_results)
    return snap


# ─── 断言式评分 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expectations", "expected_status"),
    [
        ([{"type": "no_tool_errors", "weight": 1.0}], "passed"),
        ([{"type": "output_contains", "pattern": "PASSED", "weight": 1.0}], "passed"),
        ([{"type": "output_contains", "pattern": "missing", "weight": 1.0}], "failed"),
        ([{"type": "tool_called", "tool": "bash", "args_regex": "pytest", "weight": 1.0}],
         "passed"),
        ([{"type": "tool_called", "tool": "grep", "weight": 1.0}], "failed"),
        ([{"type": "tool_call_count", "tool": "bash", "min": 1, "max": 2, "weight": 1.0}],
         "passed"),
        ([{"type": "tool_call_count", "tool": "bash", "min": 5, "weight": 1.0}], "failed"),
        ([{"type": "tool_called", "tool": "bash", "after_tool": "edit", "weight": 1.0}],
         "passed"),
    ],
    ids=lambda v: (v[0]["type"] + "-" + str(v[0].get("tool", "")))
    if isinstance(v, list)
    else str(v),
)
def test_assertion_expectations(expectations, expected_status) -> None:
    result = score_assertions(_snap(), expectations)
    assert result.status == expected_status


def test_assertion_weights_blend() -> None:
    """权重叠加：2/3 得分 → passed（阈值 0.6），1/3 → failed。"""
    expectations = [
        {"type": "no_tool_errors", "weight": 1.0},
        {"type": "tool_called", "tool": "grep", "weight": 1.0},
        {"type": "tool_called", "tool": "bash", "weight": 1.0},
    ]
    result = score_assertions(_snap(), expectations)
    assert result.score == pytest.approx(2 / 3, abs=1e-4)
    assert result.status == "passed"


def test_assertion_unknown_type_is_skipped() -> None:
    expectations = [
        {"type": "future_checker", "weight": 1.0},
        {"type": "no_tool_errors", "weight": 1.0},
    ]
    result = score_assertions(_snap(), expectations)
    assert result.score == 1.0


@pytest.mark.parametrize("expectations", [[], [{"type": "future_checker", "weight": 1.0}]])
def test_assertion_without_scorable_content_returns_none_score(expectations) -> None:
    result = score_assertions(_snap(), expectations)
    assert result.score is None
    assert result.status == "passed"


def test_tool_error_fails_no_tool_errors() -> None:
    snap = _snap(tool_results=[ToolResult(seq=3, call_id="a", output="boom", is_error=True)])
    result = score_assertions(snap, [{"type": "no_tool_errors", "weight": 1.0}])
    assert result.status == "failed"
    assert "had tool errors" in result.reason


def test_output_contains_regex_mode() -> None:
    snap = _snap(text_outputs=["error E404: not found"])
    ok = score_assertions(snap, [{"type": "output_contains", "pattern": r"E\d+",
                                  "regex": True, "weight": 1.0}])
    assert ok.status == "passed"
    miss = score_assertions(snap, [{"type": "output_contains", "pattern": r"W\d+",
                                    "regex": True, "weight": 1.0}])
    assert miss.status == "failed"


# ─── 执行式评分（真实子进程）─────────────────────────────────────────────


def _write_solution(work_dir, code: str) -> None:
    (work_dir / "solution.py").write_text(code, encoding="utf-8")


def test_execution_check_passes_on_correct_solution(tmp_path) -> None:
    _write_solution(tmp_path, "def double(x):\n    return x * 2\n")
    check = "from solution import double as candidate\nassert candidate(3) == 6\n"
    ok, output = run_execution_check(tmp_path, check)
    assert ok
    assert output == ""


def test_execution_check_fails_on_wrong_solution(tmp_path) -> None:
    _write_solution(tmp_path, "def double(x):\n    return x + 2\n")
    check = "from solution import double as candidate\nassert candidate(3) == 6\n"
    ok, output = run_execution_check(tmp_path, check)
    assert not ok
    assert "AssertionError" in output or "assert" in output


def test_execution_check_fails_when_solution_missing(tmp_path) -> None:
    check = "from solution import double\n"
    ok, output = run_execution_check(tmp_path, check)
    assert not ok
    assert "ModuleNotFoundError" in output


def test_execution_check_timeout(tmp_path) -> None:
    check = "import time\ntime.sleep(5)\n"
    ok, output = run_execution_check(tmp_path, check, timeout_s=0.2)
    assert not ok
    assert "timeout" in output


def test_execution_check_isolated_from_env_pythonpath(tmp_path, monkeypatch) -> None:
    """-I 隔离环境注入：作弊模块放 PYTHONPATH 也 import 不到。"""
    cheater = tmp_path / "cheater"
    cheater.mkdir()
    (cheater / "solution.py").write_text("def double(x):\n    return 0\n",
                                         encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(cheater))
    check = "from solution import double as candidate\nassert candidate(3) == 6\n"
    ok, _ = run_execution_check(tmp_path, check)
    assert not ok  # 命中的是 tmp_path 里的空缺 solution，而非 PYTHONPATH 的作弊模块


# ─── score_case 组合 ─────────────────────────────────────────────────────


def test_score_case_execution_overrides_assertions(tmp_path) -> None:
    """check 存在时执行式一票定制；断言明细进 details 供诊断。"""
    _write_solution(tmp_path, "def double(x):\n    return x * 2\n")
    result = score_case(
        _snap(),
        check="from solution import double as candidate\nassert candidate(3) == 6\n",
        expectations=[{"type": "no_tool_errors", "weight": 1.0}],
        work_dir=tmp_path,
    )
    assert result.status == "passed"
    assert result.score == 1.0
    assert result.details  # 断言明细保留


def test_score_case_falls_back_to_assertions_without_check(tmp_path) -> None:
    result = score_case(
        _snap(),
        check=None,
        expectations=[{"type": "no_tool_errors", "weight": 1.0}],
        work_dir=tmp_path,
    )
    assert result.status == "passed"
