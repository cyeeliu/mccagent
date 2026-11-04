"""质量门禁类型定义：检查策略、检查项与检查结果。

用可执行检查验证 Agent 交付，避免仅凭自然语言报告将任务判定为完成。
通过 QualityPolicy 描述检查策略，
通过 Check 描述单个检查项（测试命令、静态检查、格式检查等），
通过 CheckResult 描述检查执行结果（含退出码、输出、耗时）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class QualityError(RuntimeError):
    """质量门禁基础错误。"""


class CheckTimeoutError(QualityError):
    """检查执行超时。"""


class CheckNotFoundError(QualityError):
    """检查命令不存在。"""


class CheckKind(StrEnum):
    """检查类型。"""

    TEST = "test"
    LINT = "lint"
    FORMAT = "format"
    FILE_EXISTS = "file_exists"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Check:
    """单个检查项：命令、类型与约束。

    name 为检查名称（唯一标识）。
    kind 为检查类型。
    command 为执行命令（如 ["python", "-m", "pytest"]）。
    required 为是否必选（默认 True）。
    timeout 为超时秒数（None 表示不限制）。
    workdir 为工作目录（None 表示继承调用方）。
    description 为检查描述。
    """

    name: str
    kind: CheckKind
    command: tuple[str, ...]
    required: bool = True
    timeout: float | None = None
    workdir: Path | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise QualityError("name must not be empty")
        if not self.command:
            raise QualityError("command must not be empty")
        if self.timeout is not None and self.timeout <= 0:
            raise QualityError("timeout must be positive")

    @property
    def command_list(self) -> list[str]:
        """转换为 list 供 subprocess 使用。"""
        return list(self.command)


@dataclass(frozen=True)
class CheckResult:
    """检查执行结果：通过/失败 + 退出码 + 输出 + 耗时。

    name / kind 关联检查项。
    passed 为是否通过（exit_code == 0）。
    exit_code 为命令退出码。
    output 为 stdout + stderr 合并输出。
    duration_secs 为执行耗时（秒）。
    error 为错误描述（超时、命令不存在等）。
    required 为是否必选检查。
    """

    name: str
    kind: CheckKind
    passed: bool
    exit_code: int
    output: str = ""
    duration_secs: float = 0.0
    error: str | None = None
    required: bool = True

    @property
    def is_failure(self) -> bool:
        return not self.passed

    @property
    def is_required_failure(self) -> bool:
        return self.is_failure and self.required


@dataclass(frozen=True)
class QualityPolicy:
    """质量策略：检查项集合 + 重试配置。

    checks 为检查项列表。
    max_retries 为修复循环最大重试次数。
    description 为策略描述。
    """

    checks: tuple[Check, ...] = ()
    max_retries: int = 3
    description: str = ""

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise QualityError("max_retries must be >= 0")
        names = [c.name for c in self.checks]
        if len(names) != len(set(names)):
            raise QualityError("check names must be unique")

    @property
    def required_checks(self) -> tuple[Check, ...]:
        """必选检查项。"""
        return tuple(c for c in self.checks if c.required)

    @property
    def optional_checks(self) -> tuple[Check, ...]:
        """可选检查项。"""
        return tuple(c for c in self.checks if not c.required)

    @property
    def has_checks(self) -> bool:
        return len(self.checks) > 0


@dataclass(frozen=True)
class GateResult:
    """质量门禁结果：全部检查结果汇总。

    results 为各检查项的结果。
    required_passed / required_failed 为必选检查通过/失败数。
    optional_passed / optional_failed 为可选检查通过/失败数。
    attempt 为当前尝试次数（1 表示首次）。
    """

    results: tuple[CheckResult, ...] = ()
    required_passed: int = 0
    required_failed: int = 0
    optional_passed: int = 0
    optional_failed: int = 0
    attempt: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """必选检查是否全部通过。"""
        return self.required_failed == 0 and self.required_passed > 0

    @property
    def all_passed(self) -> bool:
        """全部检查（含可选）是否通过。"""
        return self.required_failed == 0 and self.optional_failed == 0

    @property
    def has_required_failures(self) -> bool:
        return self.required_failed > 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def summary(self) -> str:
        """结果摘要。"""
        return (
            f"required: {self.required_passed} passed, "
            f"{self.required_failed} failed; "
            f"optional: {self.optional_passed} passed, "
            f"{self.optional_failed} failed"
        )

    def get_result(self, name: str) -> CheckResult | None:
        """按检查名称查询结果。"""
        for r in self.results:
            if r.name == name:
                return r
        return None

    @property
    def failed_required(self) -> tuple[CheckResult, ...]:
        """失败的必选检查结果。"""
        return tuple(r for r in self.results if r.is_required_failure)
