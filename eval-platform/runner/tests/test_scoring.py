"""评分器测试：四类断言、权重、阈值与缺省行为。"""

import pytest
from runner.events import LlmChunk, ToolCall, ToolResult
from runner.scoring import TraceSnapshot, score


def _snap(
    tool_calls: list[ToolCall] | None = None,
    tool_results: list[ToolResult] | None = None,
    texts: list[str] | None = None,
    had_error: bool = False,
) -> TraceSnapshot:
    return TraceSnapshot(
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        text_outputs=texts or [],
        had_error=had_error,
    )


def _call(tool: str, args: object = None) -> ToolCall:
    return ToolCall(seq=0, turn=0, call_id="c1", tool_name=tool, args=args)


def _result(is_error: bool = False) -> ToolResult:
    return ToolResult(seq=1, call_id="c1", output="out", is_error=is_error)


def test_no_expectations_passes_with_null_score() -> None:
    result = score(_snap(), [])
    assert result.status == "passed"
    assert result.score is None
    assert "manual review" in result.reason


def test_tool_called_basic() -> None:
    snap = _snap(tool_calls=[_call("bash", {"command": "pytest -q"})])
    result = score(snap, [{"type": "tool_called", "tool": "bash", "weight": 1.0}])
    assert result.status == "passed"
    assert result.score == 1.0


def test_tool_called_missing_tool_fails() -> None:
    snap = _snap(tool_calls=[_call("grep")])
    result = score(snap, [{"type": "tool_called", "tool": "bash", "weight": 1.0}])
    assert result.status == "failed"
    assert result.score == 0.0
    assert "not called" in result.reason


def test_tool_called_args_regex() -> None:
    snap = _snap(tool_calls=[_call("bash", {"command": "cargo test"})])
    result = score(
        snap, [{"type": "tool_called", "tool": "bash", "args_regex": "pytest", "weight": 1.0}]
    )
    assert result.status == "failed"
    assert "args did not match" in result.reason

    snap2 = _snap(tool_calls=[_call("bash", {"command": "pytest -q"})])
    result2 = score(
        snap2, [{"type": "tool_called", "tool": "bash", "args_regex": "pytest", "weight": 1.0}]
    )
    assert result2.status == "passed"


def test_tool_called_ordering_constraint() -> None:
    calls = [
        _call("bash", {"command": "echo hi"}),
        _call("edit", {"path": "a.rs"}),
    ]
    snap = _snap(tool_calls=calls)
    ok = score(snap, [{"type": "tool_called", "tool": "edit", "after_tool": "bash"}])
    bad = score(snap, [{"type": "tool_called", "tool": "bash", "after_tool": "edit"}])
    assert ok.status == "passed"
    assert bad.status == "failed"
    assert "not called after" in bad.reason


@pytest.mark.parametrize(
    ("texts", "expectation", "should_pass"),
    [
        (["all tests passed"], {"type": "output_contains", "pattern": "PASSED"}, True),
        (["nothing here"], {"type": "output_contains", "pattern": "passed"}, False),
        (["error code E042"], {"type": "output_contains", "pattern": r"E\d+", "regex": True}, True),
        (["error code X"], {"type": "output_contains", "pattern": r"E\d+", "regex": True}, False),
    ],
)
def test_output_contains(
    texts: list[str], expectation: dict, should_pass: bool
) -> None:
    result = score(_snap(texts=texts), [expectation])
    assert result.status == ("passed" if should_pass else "failed")


def test_no_tool_errors() -> None:
    ok = score(_snap(tool_results=[_result(is_error=False)]), [{"type": "no_tool_errors"}])
    bad = score(_snap(had_error=True), [{"type": "no_tool_errors"}])
    assert ok.status == "passed"
    assert bad.status == "failed"
    assert "had tool errors" in bad.reason


@pytest.mark.parametrize(
    ("count", "exp", "should_pass"),
    [
        (2, {"type": "tool_call_count", "tool": "bash", "min": 1, "max": 3}, True),
        (5, {"type": "tool_call_count", "tool": "bash", "min": 1, "max": 3}, False),
        (0, {"type": "tool_call_count", "tool": "bash", "min": 0}, True),
    ],
)
def test_tool_call_count(count: int, exp: dict, should_pass: bool) -> None:
    calls = [_call("bash") for _ in range(count)]
    result = score(_snap(tool_calls=calls), [exp])
    assert result.status == ("passed" if should_pass else "failed")


def test_weighted_score_and_default_threshold() -> None:
    snap = _snap(
        tool_calls=[_call("bash", {"command": "pytest"})],
        texts=["tests passed"],
        had_error=True,
    )
    result = score(
        snap,
        [
            {"type": "tool_called", "tool": "bash", "weight": 0.4},
            {"type": "output_contains", "pattern": "passed", "weight": 0.4},
            {"type": "no_tool_errors", "weight": 0.2},
        ],
    )
    assert result.score == pytest.approx(0.8)
    assert result.status == "passed"  # 0.8 >= 默认阈值 0.6

    snap_bad = _snap(had_error=True)
    result_bad = score(
        snap_bad,
        [
            {"type": "tool_called", "tool": "bash", "weight": 0.4},
            {"type": "no_tool_errors", "weight": 0.2},
        ],
    )
    assert result_bad.score == pytest.approx(0.0)
    assert result_bad.status == "failed"


def test_threshold_override_on_first_expectation() -> None:
    snap = _snap(tool_calls=[_call("bash")])
    result = score(
        snap,
        [
            {
                "type": "tool_called",
                "tool": "bash",
                "weight": 1.0,
                "pass_threshold": 0.99,
            }
        ],
    )
    assert result.score == 1.0
    assert result.status == "passed"

    snap_no_tools = _snap()
    result2 = score(
        snap_no_tools,
        [
            {
                "type": "tool_called",
                "tool": "bash",
                "weight": 0.5,
                "pass_threshold": 0.4,
            },
            {"type": "output_contains", "pattern": "x", "weight": 0.5},
        ],
    )
    assert result2.score == 0.0
    assert result2.status == "failed"  # 0.0 < 0.4


def test_unknown_expectation_type_skipped() -> None:
    snap = _snap(tool_calls=[_call("bash")])
    result = score(
        snap,
        [
            {"type": "tool_called", "tool": "bash", "weight": 1.0},
            {"type": "custom_future_check", "weight": 5.0},
        ],
    )
    assert result.score == 1.0
    assert result.status == "passed"


def test_all_unknown_types_score_is_none() -> None:
    result = score(_snap(), [{"type": "custom_check", "weight": 1.0}])
    assert result.score is None
    assert result.status == "passed"
    assert result.reason == "no scorable expectations"


def test_reason_contains_checkmarks() -> None:
    snap = _snap(tool_calls=[_call("bash")])
    result = score(
        snap,
        [
            {"type": "tool_called", "tool": "bash", "weight": 0.5},
            {"type": "output_contains", "pattern": "missing", "weight": 0.5},
        ],
    )
    assert "✓ tool_called" in result.reason
    assert "✗ output_contains" in result.reason
