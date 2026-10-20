"""SSE 广播中心：按主题（run / batch）分发轨迹事件。

每个主题维护订阅者列表；发布是无阻塞的——慢消费者的队列写满时
丢弃事件并累计 lagged 计数，订阅端读到 lagged 信号后走 REST 全量
补齐，服务端绝不因单个慢客户端而反压事件摄取路径。
"""

from __future__ import annotations

import asyncio
from typing import NamedTuple

from eval_server.domain import TraceEvent

SUBSCRIBER_BUFFER = 1024


class Topic(NamedTuple):
    kind: str  # "run" | "batch"
    id: str

    @classmethod
    def run(cls, run_id: str) -> Topic:
        return cls("run", run_id)

    @classmethod
    def batch(cls, batch_id: str) -> Topic:
        return cls("batch", batch_id)


class Subscription:
    """单订阅者句柄：产出 ("event", TraceEvent) 或 ("lagged", 丢弃条数)。"""

    def __init__(self, hub: StreamHub, topic: Topic) -> None:
        self.queue: asyncio.Queue[TraceEvent] = asyncio.Queue(maxsize=SUBSCRIBER_BUFFER)
        self.lagged = 0
        self._hub = hub
        self._topic = topic
        self._closed = False

    async def next(self) -> tuple[str, TraceEvent | int]:
        lagged = self.lagged
        if lagged:
            self.lagged = 0
            return "lagged", lagged
        ev = await self.queue.get()
        return "event", ev

    def close(self) -> None:
        """从主题注销；已缓冲的事件仍可继续消费。"""
        if not self._closed:
            self._closed = True
            self._hub._unsubscribe(self._topic, self)


class StreamHub:
    """主题到订阅者的注册表；非线程安全，仅在事件循环内使用。"""

    def __init__(self) -> None:
        self._topics: dict[Topic, list[Subscription]] = {}

    def publish(self, topic: Topic, event: TraceEvent) -> None:
        for sub in self._topics.get(topic, []):
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                sub.lagged += 1

    def subscribe(self, topic: Topic) -> Subscription:
        sub = Subscription(self, topic)
        self._topics.setdefault(topic, []).append(sub)
        return sub

    def retire(self, topic: Topic) -> None:
        """run 结束后回收主题注册，防止注册表无限增长。"""
        self._topics.pop(topic, None)

    def subscriber_count(self, topic: Topic) -> int:
        return len(self._topics.get(topic, []))

    def _unsubscribe(self, topic: Topic, sub: Subscription) -> None:
        subs = self._topics.get(topic)
        if subs and sub in subs:
            subs.remove(sub)
            if not subs:
                self._topics.pop(topic, None)
