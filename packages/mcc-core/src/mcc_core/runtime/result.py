"""RuntimeResult 与相关类型。

所有终止路径都必须形成 RuntimeResult（迁移方案 §3.5）：
成功、失败、取消、超时、拒绝、Runtime 崩溃。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from mcc_core.runtime.enums import TaskStatus
from mcc_core.runtime.identifiers import AgentId, AttemptId, RuntimeId, TaskId


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass(frozen=True)
class RuntimeErrorInfo:
    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    mime_type: str = "application/octet-stream"
    size_bytes: int | None = None
    checksum: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class RuntimeResult:
    task_id: TaskId
    attempt_id: AttemptId
    runtime_id: RuntimeId
    agent_id: AgentId
    status: TaskStatus
    output: Mapping[str, object] = field(default_factory=dict)
    summary: str = ""
    error: RuntimeErrorInfo | None = None
    usage: TokenUsage | None = None
    cost_usd: Decimal | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifacts: tuple[ArtifactRef, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        )

    @property
    def is_success(self) -> bool:
        return self.status == TaskStatus.SUCCEEDED

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "runtime_id": self.runtime_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "output": dict(self.output),
            "summary": self.summary,
            "error": self.error.to_dict() if self.error else None,
            "usage": self.usage.to_dict() if self.usage else None,
            "cost_usd": str(self.cost_usd) if self.cost_usd is not None else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


__all__ = [
    "TokenUsage",
    "RuntimeErrorInfo",
    "ArtifactRef",
    "RuntimeResult",
]
