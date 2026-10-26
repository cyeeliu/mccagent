"""Plan approval protocol：teammate 改文件前须 lead 审批。

teammate 通过 ``submit_plan`` 提交工作计划，lead 通过 ``review_plan``
审批或驳回。审批前 teammate 的写操作被阻塞（等待 ``asyncio.Event``）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


class PlanError(Exception):
    """plan approval 错误。"""


@dataclass
class PlanRequest:
    """工作计划请求：teammate 提交，lead 审批。"""

    request_id: str
    sender: str
    plan: str
    task_id: str | None = None
    status: str = "pending"
    work_version: int = 0
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sender": self.sender,
            "plan": self.plan,
            "task_id": self.task_id,
            "status": self.status,
            "work_version": self.work_version,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }


class PlanStore:
    """plan 请求存储与审批状态机。

    ``create`` 创建 pending 请求并关联 ``asyncio.Event``；
    ``review`` 更新状态并唤醒等待的 teammate；
    ``wait_for_review`` 阻塞直到 lead 审批。
    """

    def __init__(self) -> None:
        self._requests: dict[str, PlanRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

    async def create(
        self,
        sender: str,
        plan: str,
        task_id: str | None = None,
        work_version: int = 0,
    ) -> PlanRequest:
        """创建一个 pending plan 请求。"""
        async with self._lock:
            self._counter += 1
            rid = f"plan_{self._counter:04d}"
            req = PlanRequest(
                request_id=rid,
                sender=sender,
                plan=plan,
                task_id=task_id,
                work_version=work_version,
            )
            self._requests[rid] = req
            self._events[rid] = asyncio.Event()
        return req

    async def get(self, request_id: str) -> PlanRequest | None:
        async with self._lock:
            req = self._requests.get(request_id)
            return req

    async def review(
        self,
        request_id: str,
        approve: bool,
        feedback: str = "",
    ) -> PlanRequest:
        """审批或驳回一个 plan 请求，唤醒等待的 teammate。"""
        async with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise PlanError(f"unknown plan request: {request_id}")
            if req.status != "pending":
                raise PlanError(f"plan {request_id} already {req.status}")
            req.status = "approved" if approve else "rejected"
            req.feedback = feedback
            event = self._events.get(request_id)
        if event is not None:
            event.set()
        return req

    async def wait_for_review(self, request_id: str) -> PlanRequest:
        """等待 plan 审批结果（阻塞直到 lead review）。"""
        event = self._events.get(request_id)
        if event is None:
            raise PlanError(f"unknown plan request: {request_id}")
        await event.wait()
        return self._requests[request_id]

    async def pending_for(self, sender: str) -> PlanRequest | None:
        """获取 sender 当前 pending 的 plan（如有）。"""
        async with self._lock:
            for req in self._requests.values():
                if req.sender == sender and req.status == "pending":
                    return req
        return None

    async def list_pending(self) -> list[PlanRequest]:
        """列出所有 pending plan 请求。"""
        async with self._lock:
            return [
                req for req in self._requests.values()
                if req.status == "pending"
            ]


__all__ = ["PlanError", "PlanRequest", "PlanStore"]
