"""应用状态：请求处理链路共享的全部运行时依赖。"""

from __future__ import annotations

from dataclasses import dataclass, field

from eval_server.db import DbLike
from eval_server.metrics import Metrics
from eval_server.store import TraceStoreProtocol
from eval_server.stream import StreamHub


@dataclass
class AppState:
    """一次服务实例的全部可变依赖。

    Attributes:
        db: 数据库访问接口（生产为 asyncpg 连接池，测试为内存替身）。
        traces: 轨迹存储后端。
        hub: SSE 广播中心。
        lease_ttl_s: 任务租约时长；超时未续由 reaper 回收。
        runner_secret: 内部接口 Bearer token；None 表示开发模式（放行）。
        metrics: Prometheus 指标集合。
    """

    db: DbLike
    traces: TraceStoreProtocol
    hub: StreamHub = field(default_factory=StreamHub)
    lease_ttl_s: int = 120
    runner_secret: str | None = None
    metrics: Metrics = field(default_factory=Metrics)
