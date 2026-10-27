"""可靠性测试：错误分类 / 退避 / 熔断 / 限流 / 故障注入 / AgentLoop 集成。"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from mcc_config.loader import merge
from mcc_config.models import Config, ReliabilityConfig
from mcc_core import Notice, ToolContext, Usage
from mcc_harness import (
    AgentLoopBuilder,
    CircuitBreaker,
    CircuitOpenError,
    FlakyProvider,
    RetryPolicy,
    ThrottledLlm,
    classify_llm_error,
)
from mcc_harness.reliability import InjectedFailure
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


def _http_error(status: int, retry_after: str | None = None) -> Exception:
    """构造带 duck-typed response 的异常（兼容 httpx.HTTPStatusError 语义）。"""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    err = Exception(f"http {status}")
    err.response = type(  # type: ignore[attr-defined]
        "_Resp", (), {"status_code": status, "headers": headers}
    )()
    return err


# -- classify_llm_error --


class TestClassifyLlmError:
    @pytest.mark.parametrize(
        ("error", "retryable", "retry_after_ms", "fragment"),
        [
            (_http_error(429, "7"), True, 7000, "rate limited"),
            (_http_error(429), True, None, "rate limited"),
            (_http_error(500), True, None, "server error (500)"),
            (_http_error(502), True, None, "server error (502)"),
            (_http_error(503), True, None, "server error (503)"),
            (_http_error(400), False, None, "client error (400)"),
            (_http_error(401), False, None, "client error (401)"),
            (_http_error(404), False, None, "client error (404)"),
            (TimeoutError("slow"), True, None, "network error"),
            (ConnectionError("reset"), True, None, "network error"),
            (ValueError("bug"), False, None, "non-retryable (ValueError)"),
        ],
    )
    def test_classify(
        self,
        error: BaseException,
        retryable: bool,
        retry_after_ms: int | None,
        fragment: str,
    ) -> None:
        decision = classify_llm_error(error)
        assert decision.retryable is retryable
        assert decision.retry_after_ms == retry_after_ms
        assert fragment in decision.reason

    def test_injected_failure_classified_retryable(self) -> None:
        decision = classify_llm_error(InjectedFailure(503))
        assert decision.retryable is True
        assert "503" in decision.reason

    def test_retry_after_float_header(self) -> None:
        decision = classify_llm_error(_http_error(429, "0.05"))
        assert decision.retry_after_ms == 50

    def test_retry_after_invalid_header(self) -> None:
        decision = classify_llm_error(_http_error(429, "not-a-number"))
        assert decision.retryable is True
        assert decision.retry_after_ms is None


# -- RetryPolicy --


class TestRetryPolicy:
    def test_exponential_without_jitter(self) -> None:
        policy = RetryPolicy(jitter=False)
        assert policy.delay(1) == 1.0
        assert policy.delay(2) == 2.0
        assert policy.delay(3) == 4.0

    def test_max_delay_cap(self) -> None:
        policy = RetryPolicy(jitter=False, max_delay=3.0)
        assert policy.delay(10) == 3.0

    def test_jitter_within_range(self) -> None:
        import random

        policy = RetryPolicy(jitter=True)
        rng = random.Random(42)
        for _ in range(20):
            delay = policy.delay(1, rng)
            cap = min(policy.base_delay * 2, policy.max_delay)
            assert cap / 2 <= delay <= cap

    def test_default_values(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 4
        assert policy.jitter is True


# -- CircuitBreaker --


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


class TestCircuitBreaker:
    def test_closed_below_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.check()
        assert not breaker.is_open

    def test_opens_at_threshold(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.is_open
        with pytest.raises(CircuitOpenError, match="circuit open"):
            breaker.check()

    def test_success_resets(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()
        breaker.check()
        assert not breaker.is_open

    def test_half_open_after_timeout_then_closes_on_success(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0,
                                 clock=clock)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open
        clock.advance(31.0)
        breaker.check()  # half-open 放行
        breaker.record_success()
        assert not breaker.is_open

    def test_half_open_failure_reopens(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=30.0,
                                 clock=clock)
        breaker.record_failure()
        breaker.record_failure()
        clock.advance(31.0)
        breaker.check()
        breaker.record_failure()
        assert breaker.is_open
        with pytest.raises(CircuitOpenError):
            breaker.check()


# -- ThrottledLlm --


class _EchoProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, req: Any) -> Any:
        self.calls += 1
        return "ok"

    def stream(self, req: Any):
        async def _gen():
            yield "chunk"

        return _gen()


class TestThrottledLlm:
    async def test_passthrough_unlimited(self) -> None:
        inner = _EchoProvider()
        th = ThrottledLlm(inner)
        assert await th.complete({}) == "ok"
        async for ev in th.stream({}):
            assert ev == "chunk"
        assert inner.calls == 1

    async def test_rpm_min_interval(self) -> None:
        th = ThrottledLlm(_EchoProvider(), rpm=600)  # 100ms 间隔
        t0 = time.monotonic()
        await th.complete({})
        await th.complete({})
        assert time.monotonic() - t0 >= 0.09

    async def test_concurrency_cap(self) -> None:
        inner_max = 0

        class SlowProvider(_EchoProvider):
            async def complete(self, req: Any) -> Any:
                nonlocal inner_max
                self.calls += 1
                inner_max = max(inner_max, self.calls)
                await asyncio.sleep(0.05)
                self.calls -= 1
                return "ok"

        inner = SlowProvider()
        th = ThrottledLlm(inner, max_concurrency=2)
        await asyncio.gather(*[th.complete({}) for _ in range(6)])
        assert inner_max == 2


# -- FlakyProvider --


class TestFlakyProvider:
    async def test_rate_zero_never_fails(self) -> None:
        provider = FlakyProvider(_EchoProvider(), 0.0, seed=1)
        assert await provider.complete({}) == "ok"

    async def test_rate_one_always_fails(self) -> None:
        provider = FlakyProvider(_EchoProvider(), 1.0, seed=1)
        with pytest.raises(InjectedFailure):
            await provider.complete({})
        with pytest.raises(InjectedFailure):
            provider.stream({})

    async def test_seeded_reproducible(self) -> None:
        async def first_outcome(seed: int) -> bool:
            provider = FlakyProvider(_EchoProvider(), 0.5, seed=seed)
            try:
                await provider.complete({})
                return True
            except InjectedFailure:
                return False

        assert await first_outcome(7) == await first_outcome(7)


# -- AgentLoop 集成 --


class _SequenceFailProvider:
    """前 N 次 stream 抛 503，其后转发 inner。"""

    def __init__(self, failures: int, inner: Any) -> None:
        self.failures = failures
        self.inner = inner
        self.calls = 0

    def stream(self, req: Any):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise InjectedFailure(503)
        return self.inner.stream(req)


class _RateLimitedProvider:
    """每次抛 429（带 retry-after）。"""

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, req: Any):
        self.calls += 1
        raise _http_error(429, "0.05")


class _SlowProvider:
    """每次 stream 前 sleep delay 秒。"""

    def __init__(self, inner: Any, delay: float) -> None:
        self.inner = inner
        self.delay = delay

    def stream(self, req: Any):
        async def _gen():
            await asyncio.sleep(self.delay)
            async for ev in self.inner.stream(req):
                yield ev

        return _gen()


def _make_agent(llm, ctx, **kw):
    registry = ToolRegistry()
    builder = (
        AgentLoopBuilder(llm, registry, ctx)
        .model("claude-haiku-4-5-20251001")
    )
    for method, value in kw.items():
        builder = getattr(builder, method)(value)
    return builder.build()


def _collect_notices(queue: asyncio.Queue) -> list[str]:
    out = []
    while not queue.empty():
        ev = queue.get_nowait()
        if isinstance(ev, Notice):
            out.append(ev.text)
    return out


class TestAgentLoopReliability:
    async def test_retry_recovers_from_transient_failures(
        self, ctx: ToolContext
    ) -> None:
        inner = MockLlm([single_text_turn("done")])
        provider = _SequenceFailProvider(2, inner)
        policy = RetryPolicy(jitter=False, base_delay=0.01)
        agent = _make_agent(provider, ctx, retry_policy=policy)
        queue: asyncio.Queue = asyncio.Queue()
        run = await agent.run_streaming("hi", queue)
        assert run.final_text == "done"
        assert provider.calls == 3
        notices = _collect_notices(queue)
        assert any("attempt 1/4" in n and "server error (503)" in n
                   for n in notices)

    async def test_retry_after_header_respected(self, ctx: ToolContext) -> None:
        provider = _RateLimitedProvider()
        policy = RetryPolicy(max_attempts=2, jitter=False)
        agent = _make_agent(provider, ctx, retry_policy=policy)
        queue: asyncio.Queue = asyncio.Queue()
        t0 = time.monotonic()
        with pytest.raises(Exception, match="http 429"):
            await agent.run_streaming("hi", queue)
        assert time.monotonic() - t0 >= 0.04
        assert provider.calls == 2

    async def test_non_retryable_fails_immediately(
        self, ctx: ToolContext
    ) -> None:
        class BadRequestProvider:
            calls = 0

            def stream(self, req: Any):
                self.calls += 1
                raise _http_error(400)

        provider = BadRequestProvider()
        policy = RetryPolicy(jitter=False, base_delay=0.01)
        agent = _make_agent(provider, ctx, retry_policy=policy)
        queue: asyncio.Queue = asyncio.Queue()
        with pytest.raises(Exception, match="http 400"):
            await agent.run_streaming("hi", queue)
        assert provider.calls == 1

    async def test_circuit_open_fails_fast(self, ctx: ToolContext) -> None:
        inner = MockLlm([single_text_turn("done")])
        provider = _SequenceFailProvider(0, inner)
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure()
        breaker.record_failure()
        agent = _make_agent(
            provider, ctx, retry_policy=RetryPolicy(), circuit_breaker=breaker
        )
        queue: asyncio.Queue = asyncio.Queue()
        with pytest.raises(CircuitOpenError):
            await agent.run_streaming("hi", queue)
        assert provider.calls == 0

    async def test_circuit_closes_after_recovery(self, ctx: ToolContext) -> None:
        inner = MockLlm([single_text_turn("done")])
        provider = _SequenceFailProvider(2, inner)
        breaker = CircuitBreaker(failure_threshold=3)
        policy = RetryPolicy(jitter=False, base_delay=0.01)
        agent = _make_agent(
            provider, ctx, retry_policy=policy, circuit_breaker=breaker
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert not breaker.is_open

    async def test_turn_timeout(self, ctx: ToolContext) -> None:
        from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart

        turns = [
            [
                ToolUseStart(id="t1", name="noop"),
                ToolUseInputDelta(partial_json="{}"),
                MessageStop(stop_reason="tool_use",
                            usage=Usage(input_tokens=1, output_tokens=1)),
            ],
            single_text_turn("done"),
        ]
        inner = _SlowProvider(MockLlm(turns), 0.1)
        agent = _make_agent(inner, ctx, turn_timeout=0.05)
        queue: asyncio.Queue = asyncio.Queue()
        from mcc_core.error import AgentError

        with pytest.raises(AgentError, match="turn timed out"):
            await agent.run_streaming("hi", queue)
        notices = _collect_notices(queue)
        assert any("turn timed out" in n for n in notices)


# -- Config --


class TestReliabilityConfig:
    def test_defaults(self) -> None:
        cfg = ReliabilityConfig()
        assert cfg.max_attempts == 4
        assert cfg.circuit_failure_threshold == 0
        assert cfg.rpm == 0
        assert cfg.turn_timeout_secs == 0.0

    def test_config_default_in_root(self) -> None:
        assert Config().reliability == ReliabilityConfig()

    def test_merge_project_overrides(self) -> None:
        home = Config()
        project = Config(reliability=ReliabilityConfig(
            circuit_failure_threshold=5, rpm=60,
        ))
        merged = merge(home, project)
        assert merged.reliability.circuit_failure_threshold == 5
        assert merged.reliability.rpm == 60

    def test_merge_project_default_keeps_home(self) -> None:
        home = Config(reliability=ReliabilityConfig(turn_timeout_secs=300.0))
        project = Config()
        assert merge(home, project).reliability.turn_timeout_secs == 300.0
