"""Agent 间消息总线：文件持久化的异步邮箱。

每个 recipient 拥有独立的 ``.mailboxes/{name}.jsonl`` 文件，
``send`` 追加一行 JSON、``consume`` 读取并清空（destructive read）。
``asyncio.Lock`` 按 recipient 分锁，避免不同 recipient 互相阻塞。

可靠投递协议（claim/ack/nack）：
- ``claim`` 将 pending 消息标记为 claimed 并返回，不删除文件内容；
- ``ack`` 确认处理完成并从文件中移除；
- ``nack`` 标记失败并递增 attempt，超过上限进入 dead 状态；
- ``recover`` 将 claimed 超时的消息重置为 pending（消费者崩溃恢复）。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class Message:
    """邮箱消息：发送方、接收方、内容、类型与元数据。

    可靠投递字段：
    - message_id：全局唯一标识，用于幂等去重；
    - status：pending / claimed / dead；
    - claim_token：claim 时生成的令牌，ack/nack 须匹配；
    - claimed_at：claim 时间戳，用于超时恢复；
    - attempt / max_attempts：重试计数与上限；
    - expires_at：过期时间，过期消息不被 claim；
    - priority：优先级（数值越大越优先）。
    """

    sender: str
    recipient: str
    content: str
    kind: str = "message"
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid4().hex)
    correlation_id: str | None = None
    task_id: str | None = None
    status: str = "pending"
    claim_token: str | None = None
    claimed_at: float | None = None
    attempt: int = 0
    max_attempts: int = 3
    expires_at: float | None = None
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "kind": self.kind,
            "meta": self.meta,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "task_id": self.task_id,
            "status": self.status,
            "claim_token": self.claim_token,
            "claimed_at": self.claimed_at,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "expires_at": self.expires_at,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            sender=d["sender"],
            recipient=d["recipient"],
            content=d["content"],
            kind=d.get("kind", "message"),
            meta=d.get("meta", {}),
            timestamp=d.get("timestamp", 0.0),
            message_id=d.get("message_id") or uuid4().hex,
            correlation_id=d.get("correlation_id"),
            task_id=d.get("task_id"),
            status=d.get("status", "pending"),
            claim_token=d.get("claim_token"),
            claimed_at=d.get("claimed_at"),
            attempt=d.get("attempt", 0),
            max_attempts=d.get("max_attempts", 3),
            expires_at=d.get("expires_at"),
            priority=d.get("priority", 0),
        )


class MessageBus:
    """文件持久化消息总线。

    每个 recipient 一个 jsonl 文件，``send`` 追加一行 JSON。
    可靠投递使用 claim/ack/nack 协议；``consume`` 为兼容方法，
    读取并清空全部消息（先解析再清空，跳过坏行避免丢失）。
    按 recipient 分锁，同一 recipient 的并发读写串行化。
    """

    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir / ".mailboxes"
        self._base.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, recipient: str) -> asyncio.Lock:
        lock = self._locks.get(recipient)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[recipient] = lock
        return lock

    def _path(self, recipient: str) -> Path:
        return self._base / f"{recipient}.jsonl"

    def _read_all(self, path: Path) -> list[Message]:
        """读取文件中所有消息，跳过坏行。"""
        if not path.exists() or path.stat().st_size == 0:
            return []
        text = path.read_text(encoding="utf-8")
        messages: list[Message] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                messages.append(Message.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
        return messages

    def _write_all(self, path: Path, messages: list[Message]) -> None:
        """将消息列表覆盖写入文件。"""
        if messages:
            lines = [json.dumps(msg.to_dict(), ensure_ascii=False) for msg in messages]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    @staticmethod
    def _is_expired(msg: Message, now: float) -> bool:
        return msg.expires_at is not None and now > msg.expires_at

    async def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        kind: str = "message",
        meta: dict[str, Any] | None = None,
        *,
        message_id: str | None = None,
        correlation_id: str | None = None,
        task_id: str | None = None,
        max_attempts: int = 3,
        expires_at: float | None = None,
        priority: int = 0,
    ) -> Message:
        """向 recipient 邮箱追加一条消息。

        传入 ``message_id`` 时执行幂等检查：若该 ID 已存在且非 dead，
        直接返回已有消息，不重复追加。
        """
        async with self._lock(recipient):
            path = self._path(recipient)
            if message_id is not None:
                for m in self._read_all(path):
                    if m.message_id == message_id and m.status != "dead":
                        return m
            msg = Message(
                sender=sender,
                recipient=recipient,
                content=content,
                kind=kind,
                meta=meta or {},
                message_id=message_id or uuid4().hex,
                correlation_id=correlation_id,
                task_id=task_id,
                max_attempts=max_attempts,
                expires_at=expires_at,
                priority=priority,
            )
            line = json.dumps(msg.to_dict(), ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        return msg

    async def peek(self, recipient: str) -> bool:
        """检查 recipient 是否有未读消息。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            return path.exists() and path.stat().st_size > 0

    async def consume(self, recipient: str) -> list[Message]:
        """读取并清空 recipient 的全部消息（destructive read）。

        兼容方法：先解析再清空，跳过坏行避免不可恢复丢失。
        新代码应使用 claim/ack/nack 替代。
        """
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return []
            messages = self._read_all(path)
            path.write_text("", encoding="utf-8")
        return messages

    async def claim(
        self,
        recipient: str,
        max_messages: int = 100,
        claim_timeout: float = 30.0,
    ) -> list[Message]:
        """Claim pending 消息，标记为 claimed 并返回。

        同时恢复 claimed 超时的消息（消费者崩溃恢复）。
        过期消息跳过不投递。
        """
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return []
            messages = self._read_all(path)
            now = time.time()
            claimed: list[Message] = []
            for msg in messages:
                if (
                    msg.status == "claimed"
                    and msg.claimed_at is not None
                    and now - msg.claimed_at > claim_timeout
                ):
                    msg.status = "pending"
                    msg.claim_token = None
                    msg.claimed_at = None
                if msg.status == "pending":
                    if self._is_expired(msg, now):
                        continue
                    if len(claimed) >= max_messages:
                        continue
                    msg.status = "claimed"
                    msg.claim_token = uuid4().hex
                    msg.claimed_at = now
                    claimed.append(msg)
            if claimed:
                self._write_all(path, messages)
            return claimed[:max_messages]

    async def ack(self, recipient: str, message_id: str, claim_token: str) -> bool:
        """确认消息处理完成，从文件中移除。返回是否成功。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return False
            messages = self._read_all(path)
            for i, msg in enumerate(messages):
                if msg.message_id == message_id and msg.claim_token == claim_token:
                    messages.pop(i)
                    self._write_all(path, messages)
                    return True
            return False

    async def nack(
        self,
        recipient: str,
        message_id: str,
        claim_token: str,
        reason: str = "",
    ) -> str:
        """处理失败，递增 attempt。返回 'retry' 或 'dead'。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return "dead"
            messages = self._read_all(path)
            for msg in messages:
                if msg.message_id == message_id and msg.claim_token == claim_token:
                    msg.attempt += 1
                    msg.claim_token = None
                    msg.claimed_at = None
                    if reason:
                        msg.meta.setdefault("nack_reasons", []).append(reason)
                    if msg.attempt >= msg.max_attempts:
                        msg.status = "dead"
                        self._write_all(path, messages)
                        return "dead"
                    msg.status = "pending"
                    self._write_all(path, messages)
                    return "retry"
            return "dead"

    async def recover(
        self,
        recipient: str,
        claim_timeout: float = 30.0,
    ) -> int:
        """将 claimed 超时的消息重置为 pending。返回恢复数量。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return 0
            messages = self._read_all(path)
            now = time.time()
            count = 0
            for msg in messages:
                if (
                    msg.status == "claimed"
                    and msg.claimed_at is not None
                    and now - msg.claimed_at > claim_timeout
                ):
                    msg.status = "pending"
                    msg.claim_token = None
                    msg.claimed_at = None
                    count += 1
            if count:
                self._write_all(path, messages)
            return count

    async def dead_letters(self, recipient: str) -> list[Message]:
        """查看死信队列。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return []
            messages = self._read_all(path)
            return [m for m in messages if m.status == "dead"]


__all__ = ["Message", "MessageBus"]
