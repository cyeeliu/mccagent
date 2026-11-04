"""结果收集器模块：结构化结果、产物引用、度量指标与汇总。

提供 ResultCollectorImpl 实现结果记录、查询与聚合，
兼容 ResultCollector Protocol（team.orchestrator），
可直接注入 TeamOrchestrator 替换默认 SimpleResultCollector。
"""

from __future__ import annotations

from .base import (
    AgentResult,
    ArtifactKind,
    ArtifactRef,
    ExecutionMetrics,
    ResultAlreadyExistsError,
    ResultError,
    ResultNotFoundError,
    ResultStatus,
    ResultSummary,
)
from .collector import ResultCollectorImpl

__all__ = [
    "AgentResult",
    "ArtifactKind",
    "ArtifactRef",
    "ExecutionMetrics",
    "ResultAlreadyExistsError",
    "ResultCollectorImpl",
    "ResultError",
    "ResultNotFoundError",
    "ResultStatus",
    "ResultSummary",
]
