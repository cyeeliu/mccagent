"""代码集成管理模块：变更集收集、合并、冲突解决与回滚。

提供 IntegrationManagerImpl 管理 Agent 工作区成果的集成生命周期，
支持策略化 commit/merge、冲突报告、验证回滚。
"""

from __future__ import annotations

from .base import (
    ConflictError,
    ConflictReport,
    DiffArtifact,
    IntegrationError,
    IntegrationNotFoundError,
    IntegrationRequest,
    IntegrationStatus,
    MergeResult,
    MergeStrategy,
    VerificationError,
    VerificationResult,
)
from .manager import IntegrationManagerImpl

__all__ = [
    "ConflictError",
    "ConflictReport",
    "DiffArtifact",
    "IntegrationError",
    "IntegrationManagerImpl",
    "IntegrationNotFoundError",
    "IntegrationRequest",
    "IntegrationStatus",
    "MergeResult",
    "MergeStrategy",
    "VerificationError",
    "VerificationResult",
]
