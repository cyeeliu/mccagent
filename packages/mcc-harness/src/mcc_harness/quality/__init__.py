"""质量门禁模块：可执行检查、策略评估与修复循环。

提供 QualityGateImpl 执行检查项、汇总结果、支持修复循环，
只有必选检查全部通过才可标记 COMPLETED。
"""

from __future__ import annotations

from .base import (
    Check,
    CheckKind,
    CheckNotFoundError,
    CheckResult,
    CheckTimeoutError,
    GateResult,
    QualityError,
    QualityPolicy,
)
from .gate import QualityGateImpl

__all__ = [
    "Check",
    "CheckKind",
    "CheckNotFoundError",
    "CheckResult",
    "CheckTimeoutError",
    "GateResult",
    "QualityError",
    "QualityGateImpl",
    "QualityPolicy",
]
