"""Runtime 健康检查结果。

RuntimeCoordinator 通过 health_check 收集各 Runtime 健康状态，
用于调度决策和自然退出感知。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RuntimeHealth:
    runtime_id: str
    state: HealthState
    active_tasks: int = 0
    queued_tasks: int = 0
    uptime_seconds: float = 0.0
    last_error: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.state == HealthState.HEALTHY


__all__ = [
    "HealthState",
    "RuntimeHealth",
]
