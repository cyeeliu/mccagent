"""可观测性测试：脱敏 / 凭据扫描 / JSON 日志 / 指标 / Sink / AgentLoop 埋点。"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from mcc_config.loader import merge
from mcc_config.models import Config, ObservabilityConfig, PermissionConfig
from mcc_core import ToolContext, Usage
from mcc_harness import AgentLoopBuilder, PermissionChecker
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_observability import (
    JsonFormatter,
    Metrics,
    ObservabilitySink,
    now_ms,
    redact_secrets,
    sanitize,
)
from mcc_tools import ToolRegistry


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


# -- sanitize --


class TestSanitize:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("password", "***"),
            ("api_token", "***"),
            ("API_KEY", "***"),
            ("Authorization", "***"),
            ("secret_value", "***"),
            ("apikey", "***"),
        ],
    )
    def test_sensitive_keys(self, key: str, expected: str) -> None:
        assert sanitize({key: "s3cr3t"}) == {key: expected}

    def test_normal_keys_untouched(self) -> None:
        data = {"path": "/tmp/x", "count": 3, "ok": True, "nothing": None}
        assert sanitize(data) == data

    def test_nested_structures(self) -> None:
        data = {
            "outer": {"token": "abc", "items": [{"secret": 1}, "plain"]},
        }
        out = sanitize(data)
        assert out["outer"]["token"] == "***"
        assert out["outer"]["items"][0]["secret"] == "***"
        assert out["outer"]["items"][1] == "plain"

    def test_long_string_truncated(self) -> None:
        long_text = "x" * 600
        out = sanitize({"text": long_text})
        assert out["text"].startswith("x" * 200)
        assert "[len=600]" in out["text"]
        assert len(out["text"]) < 300

    def test_short_string_untouched(self) -> None:
        assert sanitize({"text": "short"}) == {"text": "short"}

    def test_top_level_string(self) -> None:
        assert sanitize("y" * 600).endswith("[len=600]")
        assert sanitize("y" * 10) == "y" * 10

    def test_tuple_converted_to_list(self) -> None:
        assert sanitize(("a", "b")) == ["a", "b"]


# -- redact_secrets --


class TestRedactSecrets:
    @pytest.mark.parametrize(
        ("text", "needle"),
        [
            ("aws key AKIAIOSFODNN7EXAMPLE ok", "[REDACTED]"),
            ("openai sk-" + "a" * 25, "[REDACTED]"),
            ("anthropic sk-ant-" + "b" * 55, "[REDACTED]"),
            ("github ghp_" + "c" * 35, "[REDACTED]"),
            ("-----BEGIN RSA PRIVATE KEY-----", "[REDACTED]"),
            ("-----BEGIN OPENSSH PRIVATE KEY-----", "[REDACTED]"),
        ],
    )
    def test_redacts(self, text: str, needle: str) -> None:
        out = redact_secrets(text)
        assert needle in out
        assert "AKIA" not in out or "AKIA" not in needle
        assert "sk-" not in out.split("[REDACTED]")[0] or needle != "[REDACTED]"

    def test_plain_text_untouched(self) -> None:
        text = "hello world, no secrets here"
        assert redact_secrets(text) == text

    def test_multiple_patterns(self) -> None:
        text = f"a ghp_{'c' * 35} b AKIAIOSFODNN7EXAMPLE"
        out = redact_secrets(text)
        assert out.count("[REDACTED]") == 2

    def test_short_sk_prefix_not_flagged(self) -> None:
        assert redact_secrets("sk-short") == "sk-short"


# -- JsonFormatter --


class TestJsonFormatter:
    def _record(self, msg: str = "hello", **extra: Any) -> logging.LogRecord:
        record = logging.LogRecord(
            "mcc", logging.INFO, __file__, 10, msg, (), None
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_json_line_parseable(self) -> None:
        line = JsonFormatter().format(self._record())
        payload = json.loads(line)
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "mcc"
        assert "ts" in payload

    def test_extra_fields_included(self) -> None:
        line = JsonFormatter().format(
            self._record("llm", model="claude", latency_ms=12)
        )
        payload = json.loads(line)
        assert payload["model"] == "claude"
        assert payload["latency_ms"] == 12

    def test_sanitizes_extra_fields(self) -> None:
        line = JsonFormatter().format(
            self._record("x", api_key="secret-value")
        )
        payload = json.loads(line)
        assert payload["api_key"] == "***"

    def test_sanitize_disabled(self) -> None:
        line = JsonFormatter(sanitize_fields=False).format(
            self._record("x", api_key="secret-value")
        )
        payload = json.loads(line)
        assert payload["api_key"] == "secret-value"

    def test_text_mode(self) -> None:
        out = JsonFormatter(text=True).format(
            self._record("hello", tool="read_file")
        )
        assert "INFO" in out and "hello" in out and "read_file" in out
        assert not out.startswith("{")

    def test_no_extra_fields(self) -> None:
        line = JsonFormatter().format(self._record())
        json.loads(line)


# -- Metrics --


class TestMetrics:
    def test_counter_accumulates(self) -> None:
        m = Metrics()
        m.counter("c1")
        m.counter("c1", amount=2.5)
        snap = m.snapshot()
        entry = next(c for c in snap["counters"] if c["name"] == "c1")
        assert entry["value"] == 3.5

    def test_counter_label_dimensions(self) -> None:
        m = Metrics()
        m.counter("c", model="a")
        m.counter("c", model="b")
        m.counter("c", model="a")
        snap = m.snapshot()
        by_model = {
            c["labels"]["model"]: c["value"] for c in snap["counters"]
        }
        assert by_model == {"a": 2.0, "b": 1.0}

    def test_histogram_stats(self) -> None:
        m = Metrics()
        for value in [10, 20, 30, 40, 50]:
            m.histogram("lat", value, tool="t")
        snap = m.snapshot()
        entry = snap["histograms"][0]
        assert entry["count"] == 5
        assert entry["sum"] == 150
        assert entry["min"] == 10
        assert entry["max"] == 50
        assert entry["p50"] == 30
        assert entry["p95"] == 50

    def test_gauge_overwrites(self) -> None:
        m = Metrics()
        m.gauge("g", 1.0)
        m.gauge("g", 7.0)
        snap = m.snapshot()
        assert snap["gauges"][0]["value"] == 7.0

    def test_empty_snapshot(self) -> None:
        snap = Metrics().snapshot()
        assert snap == {"counters": [], "histograms": [], "gauges": []}

    def test_snapshot_json_serializable(self) -> None:
        m = Metrics()
        m.counter("c", x="y")
        m.histogram("h", 1.0)
        m.gauge("g", 2.0)
        json.dumps(m.snapshot())


# -- ObservabilitySink --


class TestObservabilitySink:
    def _make(self, **kw: Any) -> tuple[ObservabilitySink, list[logging.LogRecord]]:
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("mcc.test.sink")
        logger.handlers = [_Handler()]
        logger.setLevel(logging.INFO)
        logger.propagate = False
        sink = ObservabilitySink(logger, Metrics(), session_id="s1", **kw)
        return sink, records

    def test_llm_call_logs_and_metrics(self) -> None:
        sink, records = self._make()
        sink.llm_call(
            model="m1", latency_ms=120, input_tokens=100, output_tokens=50,
            cache_read_tokens=800, cache_write_tokens=0,
            stop_reason="end_turn",
        )
        assert len(records) == 1
        assert records[0].model == "m1"
        assert records[0].session_id == "s1"
        snap = sink.metrics.snapshot()
        names = {c["name"] for c in snap["counters"]}
        assert "agent_llm_calls_total" in names
        assert "agent_llm_tokens_total" in names
        hist_names = {h["name"] for h in snap["histograms"]}
        assert "agent_llm_latency_ms" in hist_names
        assert "agent_cache_hit_rate" in hist_names

    def test_llm_call_no_cache_skips_cache_metrics(self) -> None:
        sink, _ = self._make()
        sink.llm_call(
            model="m", latency_ms=1, input_tokens=10, output_tokens=5,
            cache_read_tokens=0, cache_write_tokens=0, stop_reason=None,
        )
        snap = sink.metrics.snapshot()
        kinds = {
            c["labels"]["kind"] for c in snap["counters"]
            if c["name"] == "agent_llm_tokens_total"
        }
        assert kinds == {"input", "output"}

    def test_tool_call(self) -> None:
        sink, records = self._make()
        sink.tool_call(tool="run_bash", latency_ms=33, is_error=False)
        assert records[0].tool == "run_bash"
        snap = sink.metrics.snapshot()
        entry = next(
            c for c in snap["counters"]
            if c["name"] == "agent_tool_calls_total"
        )
        assert entry["labels"] == {"tool": "run_bash", "is_error": "False"}

    def test_permission_and_iteration(self) -> None:
        sink, records = self._make()
        sink.permission(category="Bash", decision="allow")
        sink.iteration(iteration=3)
        assert records[0].category == "Bash"
        assert records[1].iteration == 3
        snap = sink.metrics.snapshot()
        names = {c["name"] for c in snap["counters"]}
        assert "agent_permission_decisions_total" in names
        assert "agent_loop_iterations_total" in names

    def test_sanitize_applied(self) -> None:
        sink, records = self._make()
        sink._log("test", api_key="leak", normal="ok")
        assert records[0].api_key == "***"
        assert records[0].normal == "ok"

    def test_now_ms_monotonic(self) -> None:
        a = now_ms()
        b = now_ms()
        assert b >= a


# -- AgentLoop 集成 --


class _FakeSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def iteration(self, **kw: Any) -> None:
        self.calls.append(("iteration", kw))

    def llm_call(self, **kw: Any) -> None:
        self.calls.append(("llm_call", kw))

    def tool_call(self, **kw: Any) -> None:
        self.calls.append(("tool_call", kw))

    def permission(self, **kw: Any) -> None:
        self.calls.append(("permission", kw))


class _NoOpTool:
    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> Any:
        from mcc_core import ToolOutput

        return ToolOutput.ok("ok")


def _noop_turn(tid: str) -> list:
    return [
        ToolUseStart(id=tid, name="noop"),
        ToolUseInputDelta(partial_json="{}"),
        MessageStop(stop_reason="tool_use", usage=Usage(
            input_tokens=10, output_tokens=5,
        )),
    ]


class TestAgentLoopInstrumentation:
    def _make_agent(self, llm, ctx, sink, permission=None):
        registry = ToolRegistry()
        registry.register(_NoOpTool())
        builder = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .observability(sink)
        )
        if permission is not None:
            builder = builder.permission(permission)
        return builder.build()

    async def test_iteration_and_llm_call_emitted(
        self, ctx: ToolContext
    ) -> None:
        sink = _FakeSink()
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sink)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        kinds = [k for k, _ in sink.calls]
        assert "iteration" in kinds
        assert "llm_call" in kinds
        llm_kwargs = next(kw for k, kw in sink.calls if k == "llm_call")
        assert llm_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert llm_kwargs["stop_reason"] == "end_turn"
        assert llm_kwargs["attempt"] == 1
        assert llm_kwargs["latency_ms"] >= 0

    async def test_tool_call_emitted(self, ctx: ToolContext) -> None:
        sink = _FakeSink()
        llm = MockLlm([_noop_turn("t1"), single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sink)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        tool_calls = [kw for k, kw in sink.calls if k == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "noop"
        assert tool_calls[0]["is_error"] is False
        assert tool_calls[0]["latency_ms"] >= 0

    async def test_permission_emitted(self, ctx: ToolContext) -> None:
        sink = _FakeSink()
        llm = MockLlm([_noop_turn("t1"), single_text_turn("done")])
        checker = PermissionChecker(PermissionConfig())
        agent = self._make_agent(llm, ctx, sink, permission=checker)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        perms = [kw for k, kw in sink.calls if k == "permission"]
        assert len(perms) == 1
        assert perms[0]["decision"] in ("allow", "deny", "ask")

    async def test_no_sink_regression(self, ctx: ToolContext) -> None:
        llm = MockLlm([_noop_turn("t1"), single_text_turn("done")])
        registry = ToolRegistry()
        registry.register(_NoOpTool())
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        from mcc_core.event import TurnEnd

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert any(isinstance(e, TurnEnd) for e in events)


# -- Config --


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        cfg = ObservabilityConfig()
        assert cfg.enabled is True
        assert cfg.log_format == "json"
        assert cfg.log_level == "info"
        assert cfg.sanitize is True
        assert cfg.metrics is True

    def test_otel_endpoint_removed(self) -> None:
        assert "otel_endpoint" not in ObservabilityConfig.model_fields

    def test_merge_project_overrides(self) -> None:
        home = Config()
        project = Config(
            observability=ObservabilityConfig(enabled=False, log_level="debug")
        )
        merged = merge(home, project)
        assert merged.observability.enabled is False
        assert merged.observability.log_level == "debug"

    def test_merge_project_default_keeps_home(self) -> None:
        home = Config(observability=ObservabilityConfig(log_format="text"))
        project = Config()
        assert merge(home, project).observability.log_format == "text"
