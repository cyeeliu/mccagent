"""QualityGate 质量门禁测试。

覆盖：
- 类型定义与校验（Check, QualityPolicy, CheckResult, GateResult）
- 单检查执行（成功、失败、超时、命令不存在）
- 多检查组合（必选 + 可选、全通过、部分失败）
- 策略评估（空策略、全必选、全可选）
- 修复循环（首次失败修复后通过、达到最大重试）
- is_completed 判断
- FILE_EXISTS 检查
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_harness.quality import (
    Check,
    CheckKind,
    CheckNotFoundError,
    CheckResult,
    CheckTimeoutError,
    GateResult,
    QualityError,
    QualityGateImpl,
    QualityPolicy,
)

# -- 辅助函数 --


def make_check(
    name: str = "test",
    kind: CheckKind = CheckKind.TEST,
    command: tuple[str, ...] = ("true",),
    required: bool = True,
    timeout: float | None = None,
    workdir: Path | None = None,
) -> Check:
    return Check(
        name=name,
        kind=kind,
        command=command,
        required=required,
        timeout=timeout,
        workdir=workdir,
    )


# -- Check --


class TestCheck:
    """检查项类型校验。"""

    def test_valid_check(self):
        c = make_check()
        assert c.name == "test"
        assert c.kind == CheckKind.TEST
        assert c.required is True

    def test_empty_name_raises(self):
        with pytest.raises(QualityError, match="name"):
            Check(name="", kind=CheckKind.TEST, command=("true",))

    def test_empty_command_raises(self):
        with pytest.raises(QualityError, match="command"):
            Check(name="test", kind=CheckKind.TEST, command=())

    @pytest.mark.parametrize("timeout", [0, -1, -0.5])
    def test_invalid_timeout_raises(self, timeout):
        with pytest.raises(QualityError, match="timeout"):
            Check(
                name="test", kind=CheckKind.TEST,
                command=("true",), timeout=timeout,
            )

    def test_command_list(self):
        c = Check(
            name="t", kind=CheckKind.TEST,
            command=("python3", "-c", "print(1)"),
        )
        assert c.command_list == ["python3", "-c", "print(1)"]

    def test_optional_check(self):
        c = make_check(required=False)
        assert c.required is False


# -- CheckKind --


class TestCheckKind:
    """检查类型枚举。"""

    @pytest.mark.parametrize("kind,expected", [
        (CheckKind.TEST, "test"),
        (CheckKind.LINT, "lint"),
        (CheckKind.FORMAT, "format"),
        (CheckKind.FILE_EXISTS, "file_exists"),
        (CheckKind.CUSTOM, "custom"),
    ])
    def test_kind_values(self, kind, expected):
        assert kind == expected


# -- QualityPolicy --


class TestQualityPolicy:
    """质量策略类型。"""

    def test_empty_policy(self):
        p = QualityPolicy()
        assert not p.has_checks
        assert p.required_checks == ()
        assert p.optional_checks == ()

    def test_required_and_optional(self):
        p = QualityPolicy(checks=(
            make_check("r1", required=True),
            make_check("o1", required=False),
        ))
        assert len(p.required_checks) == 1
        assert len(p.optional_checks) == 1

    def test_duplicate_names_raises(self):
        with pytest.raises(QualityError, match="unique"):
            QualityPolicy(checks=(
                make_check("dup"),
                make_check("dup"),
            ))

    def test_negative_max_retries_raises(self):
        with pytest.raises(QualityError, match="max_retries"):
            QualityPolicy(max_retries=-1)

    @pytest.mark.parametrize("retries", [0, 1, 5, 100])
    def test_valid_max_retries(self, retries):
        p = QualityPolicy(max_retries=retries)
        assert p.max_retries == retries


# -- CheckResult --


class TestCheckResult:
    """检查结果类型。"""

    def test_passed_result(self):
        r = CheckResult(
            name="t", kind=CheckKind.TEST,
            passed=True, exit_code=0,
        )
        assert not r.is_failure
        assert not r.is_required_failure

    def test_failed_required(self):
        r = CheckResult(
            name="t", kind=CheckKind.TEST,
            passed=False, exit_code=1, required=True,
        )
        assert r.is_failure
        assert r.is_required_failure

    def test_failed_optional(self):
        r = CheckResult(
            name="t", kind=CheckKind.TEST,
            passed=False, exit_code=1, required=False,
        )
        assert r.is_failure
        assert not r.is_required_failure


# -- GateResult --


class TestGateResult:
    """门禁结果类型。"""

    def test_all_required_passed(self):
        r = GateResult(required_passed=2, required_failed=0)
        assert r.passed
        assert not r.has_required_failures

    def test_required_failure_blocks(self):
        r = GateResult(required_passed=1, required_failed=1)
        assert not r.passed
        assert r.has_required_failures

    def test_no_required_checks_not_passed(self):
        r = GateResult(required_passed=0, required_failed=0)
        assert not r.passed

    def test_all_passed_with_optional(self):
        r = GateResult(
            required_passed=2, required_failed=0,
            optional_passed=1, optional_failed=0,
        )
        assert r.all_passed
        assert r.passed

    def test_optional_failure_still_passed(self):
        r = GateResult(
            required_passed=2, required_failed=0,
            optional_passed=0, optional_failed=1,
        )
        assert r.passed
        assert not r.all_passed

    def test_summary(self):
        r = GateResult(
            required_passed=3, required_failed=1,
            optional_passed=2, optional_failed=0,
        )
        s = r.summary
        assert "3 passed" in s
        assert "1 failed" in s

    def test_get_result(self):
        cr = CheckResult(
            name="check1", kind=CheckKind.TEST,
            passed=True, exit_code=0,
        )
        r = GateResult(results=(cr,))
        assert r.get_result("check1") is cr
        assert r.get_result("nonexistent") is None

    def test_failed_required_property(self):
        cr1 = CheckResult(
            name="r1", kind=CheckKind.TEST,
            passed=False, exit_code=1, required=True,
        )
        cr2 = CheckResult(
            name="o1", kind=CheckKind.TEST,
            passed=False, exit_code=1, required=False,
        )
        r = GateResult(results=(cr1, cr2))
        failed = r.failed_required
        assert len(failed) == 1
        assert failed[0].name == "r1"


# -- run_check --


class TestRunCheck:
    """单检查执行。"""

    def test_command_success(self):
        gate = QualityGateImpl()
        result = gate.run_check(make_check(command=("true",)))
        assert result.passed
        assert result.exit_code == 0
        assert result.duration_secs >= 0

    def test_command_failure(self):
        gate = QualityGateImpl()
        result = gate.run_check(make_check(command=("false",)))
        assert not result.passed
        assert result.exit_code != 0

    def test_command_with_output(self):
        gate = QualityGateImpl()
        result = gate.run_check(make_check(
            command=("python3", "-c", "print('hello')"),
        ))
        assert result.passed
        assert "hello" in result.output

    def test_command_not_found(self):
        gate = QualityGateImpl()
        result = gate.run_check(make_check(
            command=("nonexistent-cmd-xyz-123",),
        ))
        assert not result.passed
        assert result.error is not None
        assert "not found" in result.error

    def test_timeout(self):
        gate = QualityGateImpl()
        result = gate.run_check(make_check(
            command=("sleep", "10"),
            timeout=0.5,
        ))
        assert not result.passed
        assert result.error is not None
        assert "timeout" in result.error

    def test_workdir(self, tmp_path: Path):
        (tmp_path / "test.txt").write_text("x")
        gate = QualityGateImpl()
        result = gate.run_check(make_check(
            command=("test", "-f", "test.txt"),
            workdir=tmp_path,
        ))
        assert result.passed

    def test_check_with_workdir_overrides_param(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "marker.txt").write_text("x")
        gate = QualityGateImpl()
        result = gate.run_check(
            make_check(
                command=("test", "-f", "marker.txt"),
                workdir=sub,
            ),
            workdir=tmp_path,
        )
        assert result.passed


# -- evaluate --


class TestEvaluate:
    """策略评估。"""

    def test_empty_policy(self):
        gate = QualityGateImpl()
        result = gate.evaluate(QualityPolicy())
        assert not result.passed
        assert result.total == 0

    def test_all_required_pass(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("r1", command=("true",)),
            make_check("r2", command=("true",)),
        ))
        result = gate.evaluate(policy)
        assert result.passed
        assert result.required_passed == 2
        assert result.required_failed == 0

    def test_required_failure_blocks(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("r1", command=("true",)),
            make_check("r2", command=("false",)),
        ))
        result = gate.evaluate(policy)
        assert not result.passed
        assert result.required_passed == 1
        assert result.required_failed == 1

    def test_optional_failure_does_not_block(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("r1", command=("true",), required=True),
            make_check("o1", command=("false",), required=False),
        ))
        result = gate.evaluate(policy)
        assert result.passed
        assert result.optional_failed == 1

    def test_mixed_checks(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("test1", CheckKind.TEST, ("true",), required=True),
            make_check("lint1", CheckKind.LINT, ("true",), required=True),
            make_check("fmt1", CheckKind.FORMAT, ("false",), required=False),
            make_check("custom1", CheckKind.CUSTOM, ("true",), required=False),
        ))
        result = gate.evaluate(policy)
        assert result.passed
        assert result.required_passed == 2
        assert result.optional_passed == 1
        assert result.optional_failed == 1

    def test_all_fail(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("r1", command=("false",)),
            make_check("r2", command=("false",)),
        ))
        result = gate.evaluate(policy)
        assert not result.passed
        assert result.required_failed == 2

    def test_workdir_propagated(self, tmp_path: Path):
        (tmp_path / "exists.txt").write_text("x")
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check(
                "file_check", CheckKind.FILE_EXISTS,
                ("test", "-f", "exists.txt"),
            ),
        ))
        result = gate.evaluate(policy, workdir=tmp_path)
        assert result.passed


# -- evaluate_with_retries --


class TestEvaluateWithRetries:
    """修复循环。"""

    def test_no_repair_needed(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(checks=(
            make_check("r1", command=("true",)),
        ))
        result = gate.evaluate_with_retries(policy)
        assert result.passed
        assert result.attempt == 1

    def test_repair_succeeds(self, tmp_path: Path):
        marker = tmp_path / "fixed.txt"

        def repair(gate_result: GateResult) -> None:
            marker.write_text("fixed")

        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(
                Check(
                    name="file_check",
                    kind=CheckKind.FILE_EXISTS,
                    command=("test", "-f", str(marker)),
                ),
            ),
            max_retries=3,
        )
        result = gate.evaluate_with_retries(policy, repair=repair)
        assert result.passed
        assert result.attempt == 2

    def test_max_retries_exhausted(self):
        call_count = [0]

        def repair(gate_result: GateResult) -> None:
            call_count[0] += 1

        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(make_check("r1", command=("false",)),),
            max_retries=2,
        )
        result = gate.evaluate_with_retries(policy, repair=repair)
        assert not result.passed
        assert result.attempt == 3
        assert call_count[0] == 2

    def test_no_repair_function(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(make_check("r1", command=("false",)),),
            max_retries=3,
        )
        result = gate.evaluate_with_retries(policy, repair=None)
        assert not result.passed
        assert result.attempt == 1

    def test_zero_max_retries(self):
        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(make_check("r1", command=("false",)),),
            max_retries=0,
        )
        result = gate.evaluate_with_retries(
            policy, repair=lambda r: None,
        )
        assert not result.passed
        assert result.attempt == 1

    def test_repair_raises_quality_error(self):
        def bad_repair(gate_result: GateResult) -> None:
            raise RuntimeError("repair broken")

        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(make_check("r1", command=("false",)),),
            max_retries=3,
        )
        with pytest.raises(QualityError, match="repair failed"):
            gate.evaluate_with_retries(policy, repair=bad_repair)


# -- is_completed --


class TestIsCompleted:
    """完成判断。"""

    def test_completed_when_passed(self):
        gate = QualityGateImpl()
        result = GateResult(required_passed=2, required_failed=0)
        assert gate.is_completed(result)

    def test_not_completed_when_failed(self):
        gate = QualityGateImpl()
        result = GateResult(required_passed=1, required_failed=1)
        assert not gate.is_completed(result)

    def test_not_completed_no_checks(self):
        gate = QualityGateImpl()
        result = GateResult()
        assert not gate.is_completed(result)


# -- 端到端 --


class TestEndToEnd:
    """端到端场景。"""

    def test_full_pipeline_pass(self, tmp_path: Path):
        (tmp_path / "src.py").write_text("x = 1\n")
        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(
                Check(
                    name="file_exists",
                    kind=CheckKind.FILE_EXISTS,
                    command=("test", "-f", "src.py"),
                    required=True,
                ),
                Check(
                    name="syntax_check",
                    kind=CheckKind.LINT,
                    command=("python3", "-c", "compile(open('src.py').read(), 'src.py', 'exec')"),
                    required=True,
                ),
                Check(
                    name="format_check",
                    kind=CheckKind.FORMAT,
                    command=("true",),
                    required=False,
                ),
            ),
            max_retries=2,
        )
        result = gate.evaluate(policy, workdir=tmp_path)
        assert result.passed
        assert gate.is_completed(result)
        assert result.all_passed

    def test_required_failure_blocks_completion(self, tmp_path: Path):
        (tmp_path / "src.py").write_text("x = 1\n")
        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(
                Check(
                    name="file_exists",
                    kind=CheckKind.FILE_EXISTS,
                    command=("test", "-f", "src.py"),
                    required=True,
                ),
                Check(
                    name="missing_file",
                    kind=CheckKind.FILE_EXISTS,
                    command=("test", "-f", "nonexistent.py"),
                    required=True,
                ),
            ),
        )
        result = gate.evaluate(policy, workdir=tmp_path)
        assert not result.passed
        assert not gate.is_completed(result)
        assert len(result.failed_required) == 1

    def test_repair_cycle_fixes_issue(self, tmp_path: Path):
        target = tmp_path / "output.txt"

        def repair(gate_result: GateResult) -> None:
            target.write_text("generated\n")

        gate = QualityGateImpl()
        policy = QualityPolicy(
            checks=(
                Check(
                    name="output_exists",
                    kind=CheckKind.FILE_EXISTS,
                    command=("test", "-f", str(target)),
                    required=True,
                ),
            ),
            max_retries=3,
        )
        result = gate.evaluate_with_retries(policy, repair=repair)
        assert gate.is_completed(result)
        assert result.attempt == 2
