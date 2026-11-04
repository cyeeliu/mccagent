"""质量门禁实现：执行检查、汇总结果与修复循环。

用可执行检查验证 Agent 交付：
1. run_check — 执行单个检查项
2. evaluate — 执行策略中全部检查，汇总为 GateResult
3. evaluate_with_retries — 带修复循环的评估
4. is_completed — 判断是否可标记 COMPLETED

设计约束：
- 只有必选检查全部通过才能标记 COMPLETED。
- 命令执行受超时和工作区限制。
- 检查结果可追溯（含命令、退出码、输出、耗时）。
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .base import (
    Check,
    CheckResult,
    GateResult,
    QualityError,
    QualityPolicy,
)


class QualityGateImpl:
    """质量门禁：执行检查并汇总结果。

    所有命令通过 subprocess 执行，支持超时和工作目录约束。
    evaluate_with_retries 支持修复循环：失败后调用修复函数，
    重新执行检查，直到通过或达到最大重试次数。
    """

    def run_check(
        self,
        check: Check,
        workdir: Path | None = None,
    ) -> CheckResult:
        """执行单个检查项。

        workdir 优先使用 check.workdir，其次使用参数 workdir。
        超时抛 CheckTimeoutError（转为 failed 结果）。
        命令不存在抛 CheckNotFoundError（转为 failed 结果）。
        """
        cwd = check.workdir or workdir
        start = time.monotonic()

        try:
            result = subprocess.run(
                check.command_list,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=check.timeout,
            )
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start
            return CheckResult(
                name=check.name,
                kind=check.kind,
                passed=False,
                exit_code=-1,
                output=str(e),
                duration_secs=duration,
                error=f"timeout after {check.timeout}s",
                required=check.required,
            )
        except FileNotFoundError as e:
            duration = time.monotonic() - start
            return CheckResult(
                name=check.name,
                kind=check.kind,
                passed=False,
                exit_code=-1,
                output=str(e),
                duration_secs=duration,
                error=f"command not found: {check.command[0]}",
                required=check.required,
            )

        duration = time.monotonic() - start
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        return CheckResult(
            name=check.name,
            kind=check.kind,
            passed=passed,
            exit_code=result.returncode,
            output=output,
            duration_secs=duration,
            required=check.required,
        )

    def evaluate(
        self,
        policy: QualityPolicy,
        workdir: Path | None = None,
    ) -> GateResult:
        """执行策略中全部检查，汇总为 GateResult。

        空策略返回 passed=False（无必选检查通过）。
        """
        if not policy.has_checks:
            return GateResult()

        results: list[CheckResult] = []
        for check in policy.checks:
            results.append(self.run_check(check, workdir))

        return self._build_gate_result(results)

    def evaluate_with_retries(
        self,
        policy: QualityPolicy,
        workdir: Path | None = None,
        repair: Callable[[GateResult], None] | None = None,
    ) -> GateResult:
        """带修复循环的评估。

        首次执行 evaluate，若必选检查失败且 repair 函数提供，
        调用 repair 进行修复后重新评估。
        最多重试 max_retries 次。
        返回最后一次的 GateResult。
        """
        result = self.evaluate(policy, workdir)
        if result.passed or repair is None:
            return result

        for attempt in range(2, policy.max_retries + 2):
            try:
                repair(result)
            except Exception as e:
                raise QualityError(f"repair failed: {e}") from e

            result = self.evaluate(policy, workdir)
            result = GateResult(
                results=result.results,
                required_passed=result.required_passed,
                required_failed=result.required_failed,
                optional_passed=result.optional_passed,
                optional_failed=result.optional_failed,
                attempt=attempt,
                meta=result.meta,
            )

            if result.passed:
                return result

        return result

    @staticmethod
    def is_completed(gate_result: GateResult) -> bool:
        """判断是否可标记 COMPLETED。

        只有必选检查全部通过才返回 True。
        """
        return gate_result.passed

    @staticmethod
    def _build_gate_result(
        results: list[CheckResult],
        attempt: int = 1,
    ) -> GateResult:
        """从检查结果列表构建 GateResult。"""
        required_passed = sum(
            1 for r in results if r.required and r.passed
        )
        required_failed = sum(
            1 for r in results if r.required and r.is_failure
        )
        optional_passed = sum(
            1 for r in results if not r.required and r.passed
        )
        optional_failed = sum(
            1 for r in results if not r.required and r.is_failure
        )

        return GateResult(
            results=tuple(results),
            required_passed=required_passed,
            required_failed=required_failed,
            optional_passed=optional_passed,
            optional_failed=optional_failed,
            attempt=attempt,
        )
