"""Prometheus 指标定义。

每个 Metrics 实例持有独立的 CollectorRegistry，应用与测试互不污染。

指标体系：
- http_requests_total{method, path, status}         请求计数
- http_request_duration_seconds{method, path}        请求延迟直方图
- eval_runs_total{status}                            run 完成计数（核心 KPI）
- eval_queue_depth / eval_active_runs                队列深度快照
- eval_run_duration_seconds{status} / eval_run_cost_usd{status}
- trace_store_append_total / trace_store_lines_per_batch
- events_ingested_total{outcome}
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest


class Metrics:
    """指标集合与其文本导出。"""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests_total = Counter(
            "http_requests_total", "HTTP requests by method/path/status",
            ["method", "path", "status"], registry=self.registry,
        )
        self.http_request_duration = Histogram(
            "http_request_duration_seconds", "HTTP request latency",
            ["method", "path"], registry=self.registry,
        )
        self.eval_runs_total = Counter(
            "eval_runs_total", "Completed runs by status",
            ["status"], registry=self.registry,
        )
        self.eval_queue_depth = Gauge(
            "eval_queue_depth", "Runs currently queued", registry=self.registry,
        )
        self.eval_active_runs = Gauge(
            "eval_active_runs", "Runs currently leased or running",
            registry=self.registry,
        )
        self.eval_run_duration = Histogram(
            "eval_run_duration_seconds", "Run wall-clock duration",
            ["status"], registry=self.registry,
        )
        self.eval_run_cost = Histogram(
            "eval_run_cost_usd", "Run cost in USD",
            ["status"], registry=self.registry,
        )
        self.trace_store_appends = Counter(
            "trace_store_append_total", "Trace store append operations",
            registry=self.registry,
        )
        self.trace_store_lines = Histogram(
            "trace_store_lines_per_batch", "Lines per append batch",
            registry=self.registry,
        )
        self.events_ingested = Counter(
            "events_ingested_total", "Trace events ingested",
            ["outcome"], registry=self.registry,
        )

    def render(self) -> str:
        """导出 Prometheus text 格式。"""
        return generate_latest(self.registry).decode("utf-8")

    # ── 记录入口 ─────────────────────────────────────────────────────

    def http_request(self, method: str, path: str, status: int, duration_s: float) -> None:
        self.http_requests_total.labels(method, path, str(status)).inc()
        self.http_request_duration.labels(method, path).observe(duration_s)

    def run_completed(self, status: str, duration_s: float, cost_usd: float | None) -> None:
        self.eval_runs_total.labels(status).inc()
        self.eval_run_duration.labels(status).observe(duration_s)
        if cost_usd is not None:
            self.eval_run_cost.labels(status).observe(cost_usd)

    def update_queue_gauges(self, queued: int, active: int) -> None:
        self.eval_queue_depth.set(queued)
        self.eval_active_runs.set(active)

    def trace_append(self, lines: int) -> None:
        self.trace_store_appends.inc()
        self.trace_store_lines.observe(lines)

    def events_ingested_inc(self, accepted: int, rejected: int) -> None:
        self.events_ingested.labels("accepted").inc(accepted)
        self.events_ingested.labels("rejected").inc(rejected)
