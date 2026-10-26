"""Agent 间消息总线：文件持久化的异步邮箱。

每个 recipient 拥有独立的 ``.mailboxes/{name}.jsonl`` 文件，
``send`` 追加一行 JSON、``consume`` 读取并清空（destructive read）。
``asyncio.Lock`` 按 recipient 分锁，避免不同 recipient 互相阻塞。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """邮箱消息：发送方、接收方、内容、类型与元数据。"""

    sender: str
    recipient: str
    content: str
    kind: str = "message"
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "kind": self.kind,
            "meta": self.meta,
            "timestamp": self.timestamp,
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
        )


class MessageBus:
    """文件持久化消息总线。

    每个 recipient 一个 jsonl 文件，``send`` 追加一行 JSON，
    ``consume`` 读取全部行后清空文件。按 recipient 分锁，
    同一 recipient 的并发读写串行化，不同 recipient 互不阻塞。
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

    async def send(
        self,
        sender: str,
        recipient: str,
        content: str,
        kind: str = "message",
        meta: dict[str, Any] | None = None,
    ) -> Message:
        """向 recipient 邮箱追加一条消息。"""
        msg = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            kind=kind,
            meta=meta or {},
        )
        line = json.dumps(msg.to_dict(), ensure_ascii=False) + "\n"
        async with self._lock(recipient):
            with self._path(recipient).open("a", encoding="utf-8") as f:
                f.write(line)
        return msg

    async def peek(self, recipient: str) -> bool:
        """检查 recipient 是否有未读消息。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            return path.exists() and path.stat().st_size > 0

    async def consume(self, recipient: str) -> list[Message]:
        """读取并清空 recipient 的全部消息（destructive read）。"""
        async with self._lock(recipient):
            path = self._path(recipient)
            if not path.exists() or path.stat().st_size == 0:
                return []
            text = path.read_text(encoding="utf-8")
            path.write_text("", encoding="utf-8")
        return [
            Message.from_dict(json.loads(line))
            for line in text.splitlines()
            if line.strip()
        ]


__all__ = ["Message", "MessageBus"]
