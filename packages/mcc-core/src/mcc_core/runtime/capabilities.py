"""Runtime 能力声明。

RuntimeCapabilities 描述 Runtime 支持的操作和资源限制，
调用方通过 capabilities 属性判断是否支持特定操作，
而非 hasattr 探测。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_pause: bool = False
    supports_resume: bool = False
    supports_cancel_task: bool = True
    supports_health_check: bool = True
    supports_subscribe_events: bool = True
    supports_multiple_tasks: bool = False
    max_concurrency: int = 1
    queue_capacity: int | None = None
    max_task_timeout_seconds: float | None = None

    def can(self, operation: str) -> bool:
        return {
            "pause": self.supports_pause,
            "resume": self.supports_resume,
            "cancel_task": self.supports_cancel_task,
            "health_check": self.supports_health_check,
            "subscribe_events": self.supports_subscribe_events,
            "multiple_tasks": self.supports_multiple_tasks,
        }.get(operation, False)


INTERACTIVE_CAPABILITIES = RuntimeCapabilities(
    supports_pause=False,
    supports_resume=False,
    supports_cancel_task=True,
    supports_health_check=True,
    supports_subscribe_events=True,
    supports_multiple_tasks=False,
    max_concurrency=1,
)

TASK_CAPABILITIES = RuntimeCapabilities(
    supports_pause=False,
    supports_resume=False,
    supports_cancel_task=True,
    supports_health_check=True,
    supports_subscribe_events=True,
    supports_multiple_tasks=False,
    max_concurrency=1,
)

PERSISTENT_CAPABILITIES = RuntimeCapabilities(
    supports_pause=True,
    supports_resume=True,
    supports_cancel_task=True,
    supports_health_check=True,
    supports_subscribe_events=True,
    supports_multiple_tasks=True,
    max_concurrency=1,
    queue_capacity=32,
)


__all__ = [
    "RuntimeCapabilities",
    "INTERACTIVE_CAPABILITIES",
    "TASK_CAPABILITIES",
    "PERSISTENT_CAPABILITIES",
]
