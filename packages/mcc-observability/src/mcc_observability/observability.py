"""可观测性：结构化 JSON 日志、脱敏与进程内指标聚合。

四支柱落地方式：Logs（JsonFormatter + 文件 JSONL）、Metrics
（进程内 Metrics + 退出快照）、Replays（mcc-session SessionRecorder，
本包不重复）、成本归因（mcc-harness TurnCost，本包不重复）。
分布式 Traces（OTel）对单机 CLI 过重，明确不做。

关键约定：所有日志字段先经 ``sanitize`` 脱敏再落盘；``redact_secrets``
提供凭据正则扫描，日志与最终输出扫描共用。
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any

# -- 敏感字段脱敏 --

_SENSITIVE_KEYS = ("password", "secret", "token", "key", "authorization", "apikey")
_MAX_STR = 500
_STR_HEAD = 200

# 凭据正则：先长后短，避免 sk- 前缀吃掉 sk-ant-
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{50,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"ghp_[A-Za-z0-9]{30,}")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")),
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
]


def find_secrets(text: str) -> list[str]:
    """扫描文本中的常见凭据，返回命中模式名列表（去重，保序）。"""
    seen: list[str] = []
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text) and name not in seen:
            seen.append(name)
    return seen


def sanitize(value: Any, max_str: int = _MAX_STR) -> Any:
    """递归脱敏：超长字符串截断；敏感字段名的值置 ``***``。"""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SENSITIVE_KEYS):
                out[key] = "***"
            else:
                out[key] = sanitize(item, max_str)
        return out
    if isinstance(value, list):
        return [sanitize(item, max_str) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item, max_str) for item in value]
    if isinstance(value, str) and len(value) > max_str:
        return f"{value[:_STR_HEAD]}…[len={len(value)}]"
    return value


def redact_secrets(text: str) -> str:
    """扫描文本中的常见凭据并替换为 ``[REDACTED]``。"""
    out = text
    for _name, pattern in SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


# -- 结构化日志 --


class JsonFormatter(logging.Formatter):
    """JSON 行格式化：标准字段 + record 的额外字段（先脱敏）。

    ``text`` 模式输出人类可读单行，供开发调试。
    """

    def __init__(self, *, text: bool = False, sanitize_fields: bool = True) -> None:
        super().__init__()
        self.text = text
        self.sanitize_fields = sanitize_fields

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        if self.text:
            extra = {
                k: v
                for k, v in record.__dict__.items()
                if k not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__
            }
            suffix = f" {json.dumps(extra, ensure_ascii=False, default=str)}" if extra else ""
            return f"{ts} {record.levelname} {record.name}: {record.getMessage()}{suffix}"
        record_dict = dict(record.__dict__)
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record_dict.items():
            if key in payload or key in (
                "args", "msg", "exc_info", "exc_text", "stack_info",
                "created", "msecs", "relativeCreated", "levelname",
                "levelno", "pathname", "filename", "module", "lineno",
                "funcName", "process", "processName", "thread",
                "threadName", "name", "taskName",
            ):
                continue
            payload[key] = value
        if self.sanitize_fields:
            payload = sanitize(payload)
        return json.dumps(payload, ensure_ascii=False, default=str)


# -- 进程内指标 --


def _label_key(labels: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


class Metrics:
    """进程内指标聚合：counter / histogram / gauge。

    无外部依赖；``snapshot`` 输出可 JSON 序列化的聚合结果，
    供退出时落盘或诊断。
    """

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple], float] = {}
        self._histograms: dict[tuple[str, tuple], list[float]] = {}
        self._gauges: dict[tuple[str, tuple], float] = {}

    def counter(self, name: str, amount: float = 1.0, **labels: Any) -> None:
        key = (name, _label_key(labels))
        self._counters[key] = self._counters.get(key, 0.0) + amount

    def histogram(self, name: str, value: float, **labels: Any) -> None:
        key = (name, _label_key(labels))
        self._histograms.setdefault(key, []).append(value)

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        self._gauges[(name, _label_key(labels))] = value

    def snapshot(self) -> dict[str, Any]:
        counters = [
            {"name": name, "labels": dict(labels), "value": value}
            for (name, labels), value in sorted(self._counters.items())
        ]
        histograms = []
        for (name, labels), values in sorted(self._histograms.items()):
            histograms.append({
                "name": name,
                "labels": dict(labels),
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "p50": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
            })
        gauges = [
            {"name": name, "labels": dict(labels), "value": value}
            for (name, labels), value in sorted(self._gauges.items())
        ]
        return {"counters": counters, "histograms": histograms, "gauges": gauges}


# -- Sink 门面 --


class ObservabilitySink:
    """AgentLoop 埋点门面：一处注入，日志与指标同时记录。

    所有方法幂等、无阻塞；字段先经 ``sanitize`` 脱敏。
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        metrics: Metrics | None = None,
        *,
        session_id: str = "",
        sanitize_enabled: bool = True,
    ) -> None:
        self._logger = logger or logging.getLogger("mcc")
        self._metrics = metrics or Metrics()
        self._session_id = session_id
        self._sanitize = sanitize_enabled

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def llm_call(
        self,
        *,
        model: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        stop_reason: str | None,
        attempt: int = 1,
    ) -> None:
        """记录一次 LLM 调用（含缓存命中率指标）。"""
        self._log(
            "llm call completed",
            session_id=self._session_id,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            stop_reason=stop_reason,
            attempt=attempt,
        )
        m = self._metrics
        m.counter("agent_llm_calls_total", model=model,
                  stop_reason=stop_reason or "unknown")
        m.counter("agent_llm_tokens_total", model=model, kind="input",
                  amount=float(input_tokens))
        m.counter("agent_llm_tokens_total", model=model, kind="output",
                  amount=float(output_tokens))
        if cache_read_tokens:
            m.counter("agent_llm_tokens_total", model=model, kind="cache_read",
                      amount=float(cache_read_tokens))
        if cache_write_tokens:
            m.counter("agent_llm_tokens_total", model=model, kind="cache_write",
                      amount=float(cache_write_tokens))
        m.histogram("agent_llm_latency_ms", float(latency_ms), model=model)
        total_in = cache_read_tokens + input_tokens
        if total_in > 0:
            m.histogram(
                "agent_cache_hit_rate", cache_read_tokens / total_in, model=model
            )

    def tool_call(
        self, *, tool: str, latency_ms: int, is_error: bool
    ) -> None:
        """记录一次工具执行。"""
        self._log(
            "tool call completed",
            session_id=self._session_id,
            tool=tool,
            latency_ms=latency_ms,
            is_error=is_error,
        )
        self._metrics.counter(
            "agent_tool_calls_total", tool=tool, is_error=is_error
        )
        self._metrics.histogram("agent_tool_duration_ms", float(latency_ms), tool=tool)

    def permission(self, *, category: str, decision: str) -> None:
        """记录一次权限决策。"""
        self._log(
            "permission decision",
            session_id=self._session_id,
            category=category,
            decision=decision,
        )
        self._metrics.counter(
            "agent_permission_decisions_total", category=category, decision=decision
        )

    def iteration(self, *, iteration: int) -> None:
        """记录 agent 主循环一轮迭代。"""
        self._log(
            "loop iteration",
            session_id=self._session_id,
            iteration=iteration,
        )
        self._metrics.counter("agent_loop_iterations_total")
        self._metrics.histogram(
            "agent_loop_iterations", float(iteration),
            session_hash=str(hash(self._session_id) % 10000),
        )

    def event_counter(self, name: str, **labels: Any) -> None:
        """通用计数器：供安全告警等自定义事件聚合。"""
        self._metrics.counter(name, **labels)

    def _log(self, message: str, **fields: Any) -> None:
        if self._sanitize:
            fields = sanitize(fields)
        self._logger.info(message, extra=fields)


def now_ms() -> int:
    """埋点计时辅助：当前毫秒时间戳。"""
    return time.monotonic_ns() // 1_000_000
