"""结果收集器类型定义：执行结果、产物引用与度量指标。

将 Agent 执行产出的结构化结果集中定义，
与 TaskStore、AgentRegistry 解耦。
通过 AgentResult 描述单个 Agent 的执行结果，
通过 ArtifactRef 引用文件产物（不直接塞入消息载荷），
通过 ExecutionMetrics 记录 usage、成本和时间信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ResultError(RuntimeError):
    """结果收集器基础错误。"""


class ResultNotFoundError(ResultError):
    """结果不存在。"""


class ResultAlreadyExistsError(ResultError):
    """结果已存在，不可重复记录。"""


class ResultStatus(StrEnum):
    """单个 Agent 执行结果状态。"""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


_FAILURE_STATUSES = frozenset({
    ResultStatus.FAILED,
    ResultStatus.TIMEOUT,
    ResultStatus.CANCELLED,
    ResultStatus.BUDGET_EXCEEDED,
})


class ArtifactKind(StrEnum):
    """产物类型。"""

    FILE = "file"
    DIFF = "diff"
    LOG = "log"
    METRIC = "metric"
    OTHER = "other"


@dataclass(frozen=True)
class ArtifactRef:
    """文件产物引用：指向外部文件，不内联内容。

    path 为文件路径（相对于 worktree 或绝对路径）。
    kind 为产物类型，用于分类检索。
    checksum 为 sha256 十六进制摘要（可选）。
    size_bytes 为文件大小（可选）。
    """

    path: str
    kind: ArtifactKind = ArtifactKind.FILE
    description: str = ""
    size_bytes: int | None = None
    checksum: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path:
            raise ResultError("path must not be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ResultError("size_bytes must be >= 0")


@dataclass(frozen=True)
class ExecutionMetrics:
    """执行度量：token 用量、成本与时间。

    started_at / finished_at 为 Unix 时间戳（秒）。
    cost_usd 为本次执行的花费（美元）。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ResultError("input_tokens must be >= 0")
        if self.output_tokens < 0:
            raise ResultError("output_tokens must be >= 0")
        if self.cache_read_tokens < 0:
            raise ResultError("cache_read_tokens must be >= 0")
        if self.cache_write_tokens < 0:
            raise ResultError("cache_write_tokens must be >= 0")
        if self.cost_usd < 0:
            raise ResultError("cost_usd must be >= 0")

    @property
    def total_tokens(self) -> int:
        """输入 + 输出 token 总数。"""
        return self.input_tokens + self.output_tokens

    @property
    def duration_secs(self) -> float | None:
        """执行耗时（秒），未设置起止时间则返回 None。"""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at


@dataclass(frozen=True)
class AgentResult:
    """单个 Agent 执行结果：结构化状态 + 度量 + 产物引用。

    agent_id / task_id 关联执行主体。
    status 为结构化结果状态，不依赖自然语言 summary。
    metrics 记录 token、成本和时间。
    artifacts 为文件产物引用列表（大文件用引用不内联）。
    team_run_id / trace_id 用于跨组件关联。
    meta 存放扩展元数据（如验收检查结果）。
    """

    agent_id: str
    task_id: str
    status: ResultStatus
    summary: str = ""
    error: str | None = None
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    artifacts: tuple[ArtifactRef, ...] = ()
    team_run_id: str | None = None
    trace_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.SUCCESS

    @property
    def is_partial(self) -> bool:
        return self.status == ResultStatus.PARTIAL

    @property
    def is_failure(self) -> bool:
        return self.status in _FAILURE_STATUSES

    @property
    def has_artifacts(self) -> bool:
        return len(self.artifacts) > 0


@dataclass(frozen=True)
class ResultSummary:
    """TeamRun 结果汇总：聚合多个 AgentResult。

    run_id 关联 TeamRun。
    results 为全部 AgentResult 快照。
    """

    run_id: str
    results: tuple[AgentResult, ...] = ()

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.is_success)

    @property
    def partial(self) -> int:
        return sum(1 for r in self.results if r.is_partial)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.is_failure)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.succeeded / self.total

    @property
    def total_cost_usd(self) -> float:
        return sum(r.metrics.cost_usd for r in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.metrics.total_tokens for r in self.results)

    @property
    def has_failures(self) -> bool:
        return self.failed > 0

    @property
    def all_succeeded(self) -> bool:
        return self.total > 0 and self.succeeded == self.total

    @property
    def has_partial(self) -> bool:
        return self.partial > 0
