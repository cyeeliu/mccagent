"""HTTP 接口契约：公开 / 内部 / 系统三组端点清单。

本清单是 API 的单一事实来源：路由实现、前端客户端与文档均以此为准，
新增或变更端点时先改这里，再同步三端。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointSpec:
    method: str
    path: str
    summary: str


PUBLIC_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("POST", "/api/profiles", "创建 Agent 配置"),
    EndpointSpec("GET", "/api/profiles", "列出所有配置"),
    EndpointSpec("POST", "/api/batches", "提交批次（支持幂等键）"),
    EndpointSpec("GET", "/api/batches", "列出批次（含通过率/成本汇总）"),
    EndpointSpec("GET", "/api/batches/{id}", "批次详情"),
    EndpointSpec("POST", "/api/batches/{id}/cancel", "取消批次"),
    EndpointSpec("GET", "/api/runs", "查运行列表（过滤、分页）"),
    EndpointSpec("GET", "/api/runs/{id}", "单次运行详情"),
    EndpointSpec("GET", "/api/runs/{id}/trace", "轨迹分页读取"),
    EndpointSpec("GET", "/api/runs/{id}/stream", "SSE 实时事件流"),
    EndpointSpec("GET", "/api/reports/compare", "双批次对比报告"),
    EndpointSpec("GET", "/api/stats/dashboard", "仪表盘摘要统计"),
    EndpointSpec("GET", "/api/stats/trend", "通过率趋势"),
)

INTERNAL_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("POST", "/internal/lease", "Runner 领取任务"),
    EndpointSpec("POST", "/internal/runs/{id}/heartbeat", "心跳续租"),
    EndpointSpec("POST", "/internal/runs/{id}/events", "批量上报轨迹事件（JSONL）"),
    EndpointSpec("POST", "/internal/runs/{id}/complete", "标记运行完成"),
)

SYSTEM_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("GET", "/healthz/live", "存活探针"),
    EndpointSpec("GET", "/healthz/ready", "就绪探针（检查数据库与对象存储）"),
    EndpointSpec("GET", "/metrics", "Prometheus 文本格式指标"),
)


def all_endpoints() -> tuple[EndpointSpec, ...]:
    """三组端点的拼接视图。"""
    return PUBLIC_ENDPOINTS + INTERNAL_ENDPOINTS + SYSTEM_ENDPOINTS
