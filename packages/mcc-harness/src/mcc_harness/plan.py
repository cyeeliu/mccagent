"""Plan approval protocol：teammate 改文件前须 lead 审批。

teammate 通过 ``submit_plan`` 提交工作计划，lead 通过 ``review_plan``
审批或驳回。审批前 teammate 的写操作被阻塞（等待 ``asyncio.Event``）。

支持审批超时（deadline/timeout）、取消、幂等重复审批和文件持久化
（进程重启后恢复未完成审批）。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class PlanError(Exception):
    """plan approval 错误。"""


class PlanStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class PlanRequest:
    """工作计划请求：teammate 提交，lead 审批。

    审批元数据：
    - deadline：审批截止时间戳，超时后进入 EXPIRED；
    - approved_by / approved_at：审批人与审批时间；
    - plan_version：计划版本号，修订时递增；
    - worktree_id：关联的 worktree 标识。
    """

    request_id: str
    sender: str
    plan: str
    task_id: str | None = None
    status: str = PlanStatus.PENDING
    work_version: int = 0
    feedback: str = ""
    timestamp: float = field(default_factory=time.time)
    deadline: float | None = None
    approved_by: str | None = None
    approved_at: float | None = None
    plan_version: int = 1
    worktree_id: str | None = None

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
            "deadline": self.deadline,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "plan_version": self.plan_version,
            "worktree_id": self.worktree_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlanRequest:
        return cls(
            request_id=d["request_id"],
            sender=d["sender"],
            plan=d["plan"],
            task_id=d.get("task_id"),
            status=d.get("status", PlanStatus.PENDING),
            work_version=d.get("work_version", 0),
            feedback=d.get("feedback", ""),
            timestamp=d.get("timestamp", 0.0),
            deadline=d.get("deadline"),
            approved_by=d.get("approved_by"),
            approved_at=d.get("approved_at"),
            plan_version=d.get("plan_version", 1),
            worktree_id=d.get("worktree_id"),
        )


class PlanStore:
    """plan 请求存储与审批状态机。

    ``create`` 创建 pending 请求并关联 ``asyncio.Event``；
    ``review`` 更新状态并唤醒等待的 teammate；
    ``wait_for_review`` 阻塞直到 lead 审批或超时；
    ``cancel`` 取消未审批请求；
    ``expire_overdue`` 将超时请求标记为 EXPIRED。

    传入 ``base_dir`` 时启用文件持久化，进程重启后恢复未完成审批。
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir
        self._requests: dict[str, PlanRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._counter = 0
        if base_dir is not None:
            self._plans_dir = base_dir / ".plans"
            self._plans_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        path = self._plans_dir / "plans.json"
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("requests", []):
            req = PlanRequest.from_dict(item)
            self._requests[req.request_id] = req
            event = asyncio.Event()
            if req.status != PlanStatus.PENDING:
                event.set()
            self._events[req.request_id] = event
        self._counter = data.get("counter", len(self._requests))

    def _save(self) -> None:
        if self._base_dir is None:
            return
        path = self._plans_dir / "plans.json"
        data = {
            "counter": self._counter,
            "requests": [req.to_dict() for req in self._requests.values()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def create(
        self,
        sender: str,
        plan: str,
        task_id: str | None = None,
        work_version: int = 0,
        deadline: float | None = None,
        worktree_id: str | None = None,
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
                deadline=deadline,
                worktree_id=worktree_id,
            )
            self._requests[rid] = req
            self._events[rid] = asyncio.Event()
            self._save()
        return req

    async def get(self, request_id: str) -> PlanRequest | None:
        async with self._lock:
            return self._requests.get(request_id)

    async def review(
        self,
        request_id: str,
        approve: bool,
        feedback: str = "",
        approved_by: str | None = None,
    ) -> PlanRequest:
        """审批或驳回一个 plan 请求，唤醒等待的 teammate。

        幂等：对已审批请求重复相同结果的审批直接返回，不抛异常；
        不同结果的重复审批抛 PlanError。
        """
        new_status = PlanStatus.APPROVED if approve else PlanStatus.REJECTED
        async with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise PlanError(f"unknown plan request: {request_id}")
            if req.status == new_status:
                return req
            if req.status != PlanStatus.PENDING:
                raise PlanError(f"plan {request_id} already {req.status}")
            req.status = new_status
            req.feedback = feedback
            req.approved_by = approved_by
            req.approved_at = time.time()
            event = self._events.get(request_id)
            self._save()
        if event is not None:
            event.set()
        return req

    async def wait_for_review(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> PlanRequest:
        """等待 plan 审批结果。

        timeout 显式指定等待秒数；未指定时使用请求 deadline 计算剩余时间；
        两者均无则无限等待（兼容行为）。超时后请求进入 EXPIRED 并抛 PlanError。
        """
        event = self._events.get(request_id)
        if event is None:
            raise PlanError(f"unknown plan request: {request_id}")
        req = self._requests.get(request_id)
        effective_timeout = timeout
        if effective_timeout is None and req is not None and req.deadline is not None:
            effective_timeout = max(0.0, req.deadline - time.time())
        if effective_timeout is not None:
            try:
                await asyncio.wait_for(event.wait(), effective_timeout)
            except TimeoutError:
                async with self._lock:
                    r = self._requests.get(request_id)
                    if r is not None and r.status == PlanStatus.PENDING:
                        r.status = PlanStatus.EXPIRED
                        self._save()
                raise PlanError(f"plan {request_id} review timed out") from None
        else:
            await event.wait()
        return self._requests[request_id]

    async def cancel(self, request_id: str, reason: str = "") -> PlanRequest:
        """取消一个 pending plan 请求。"""
        async with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                raise PlanError(f"unknown plan request: {request_id}")
            if req.status != PlanStatus.PENDING:
                raise PlanError(f"plan {request_id} already {req.status}")
            req.status = PlanStatus.CANCELLED
            req.feedback = reason
            event = self._events.get(request_id)
            self._save()
        if event is not None:
            event.set()
        return req

    async def expire_overdue(self) -> list[PlanRequest]:
        """将所有超时的 pending 请求标记为 EXPIRED，返回过期列表。"""
        now = time.time()
        expired: list[PlanRequest] = []
        async with self._lock:
            for req in self._requests.values():
                if (
                    req.status == PlanStatus.PENDING
                    and req.deadline is not None
                    and now > req.deadline
                ):
                    req.status = PlanStatus.EXPIRED
                    expired.append(req)
                    event = self._events.get(req.request_id)
                    if event is not None:
                        event.set()
            if expired:
                self._save()
        return expired

    async def pending_for(self, sender: str) -> PlanRequest | None:
        """获取 sender 当前 pending 的 plan（如有）。"""
        async with self._lock:
            for req in self._requests.values():
                if req.sender == sender and req.status == PlanStatus.PENDING:
                    return req
        return None

    async def list_pending(self) -> list[PlanRequest]:
        """列出所有 pending plan 请求。"""
        async with self._lock:
            return [
                req for req in self._requests.values()
                if req.status == PlanStatus.PENDING
            ]


__all__ = ["PlanError", "PlanRequest", "PlanStatus", "PlanStore"]
