"""可靠性：结构化错误分类、退避重试、熔断、限流与故障注入。

- ``classify_llm_error`` 用 duck-typing 读取异常的 ``response.status_code``
  与 ``headers``（兼容 httpx.HTTPStatusError 与测试注入异常），
  替代脆弱的字符串匹配；429 尊重 ``retry-after`` 头。
- ``RetryPolicy`` 指数退避 + equal jitter（避免惊群）。
- ``CircuitBreaker`` 三态（closed/open/half-open），open 时快速失败。
- ``ThrottledLlm`` 包装 provider：并发信号量 + RPM 最小间隔。
- ``FlakyProvider`` 故障注入，开发期验证重试行为。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mcc_core.error import AgentError

# -- 错误分类 --


@dataclass
class RetryDecision:
    """一次 LLM 调用失败的重试裁决。"""

    retryable: bool
    retry_after_ms: int | None
    reason: str


def _retry_after_ms(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return int(float(raw) * 1000)
    except (TypeError, ValueError):
        return None


def classify_llm_error(e: BaseException) -> RetryDecision:
    """结构化错误分类：决定可重试性与建议等待时长。

    分类准则：429 可重试且尊重 retry-after；5xx 可重试；
    网络超时/连接错误可重试；其余 4xx 与未知异常不可重试。
    """
    response = getattr(e, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        if status == 429:
            return RetryDecision(True, _retry_after_ms(response), "rate limited (429)")
        if 500 <= status < 600:
            return RetryDecision(True, None, f"server error ({status})")
        return RetryDecision(False, None, f"client error ({status})")
    if isinstance(e, (TimeoutError, ConnectionError)):
        return RetryDecision(True, None, f"network error ({type(e).__name__})")
    name = type(e).__name__
    if name in ("ConnectError", "ConnectTimeout", "ReadTimeout", "PoolTimeout"):
        return RetryDecision(True, None, f"network error ({name})")
    return RetryDecision(False, None, f"non-retryable ({name})")


# -- 退避策略 --


@dataclass
class RetryPolicy:
    """指数退避 + equal jitter：delay = base*2^attempt，封顶 max_delay，
    jitter 开启时取 [cap/2, cap] 均匀随机。"""

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: bool = True

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        cap = min(self.base_delay * (2 ** min(attempt, 10)), self.max_delay)
        if not self.jitter:
            return cap
        r = rng if rng is not None else random.Random()
        return cap / 2 + r.uniform(0, cap / 2)


# -- 熔断器 --


class CircuitOpenError(AgentError):
    """熔断器 open：快速失败，给下游喘息。"""


class CircuitBreaker:
    """三态熔断器：closed →（连续失败达阈值）→ open →
    （reset_timeout 后）→ half-open（放行一次探测）→ 成功 closed /
    失败重开。asyncio 单线程访问，无需原子操作。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        clock: Any = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock if clock is not None else time.monotonic
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open = False

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def check(self) -> None:
        """放行或抛 CircuitOpenError。超时后进入 half-open 放行一次。"""
        if self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self.reset_timeout:
            self._half_open = True
            return
        raise CircuitOpenError(
            f"circuit open (failures={self._failures}, "
            f"retry after {self.reset_timeout}s)"
        )

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open = False

    def record_failure(self) -> None:
        if self._half_open:
            self._opened_at = self._clock()
            self._half_open = False
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()


# -- 限流 --


class ThrottledLlm:
    """provider 包装：并发信号量 + RPM 最小发送间隔。

    避免我方主动打爆 API；rpm/max_concurrency 为 0 表示该维度不限。
    """

    def __init__(
        self,
        inner: Any,
        rpm: int = 0,
        max_concurrency: int = 0,
    ) -> None:
        self._inner = inner
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._sem = (
            asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
        )
        self._lock = asyncio.Lock()
        self._last_sent = 0.0

    async def complete(self, req: Any) -> Any:
        async with self._acquire():
            return await self._inner.complete(req)

    def stream(self, req: Any) -> AsyncIterator:
        async def _gen() -> AsyncIterator:
            async with self._acquire():
                async for ev in self._inner.stream(req):
                    yield ev

        return _gen()

    def _acquire(self) -> _ThrottleGuard:
        return _ThrottleGuard(self)


class _ThrottleGuard:
    def __init__(self, th: ThrottledLlm) -> None:
        self._th = th
        self._sem_acquired = False

    async def __aenter__(self) -> None:
        th = self._th
        if th._sem is not None:
            await th._sem.acquire()
            self._sem_acquired = True
        if th._min_interval > 0:
            async with th._lock:
                now = time.monotonic()
                wait = th._last_sent + th._min_interval - now
                if wait > 0:
                    await asyncio.sleep(wait)
                th._last_sent = time.monotonic()

    async def __aexit__(self, *exc: Any) -> None:
        if self._sem_acquired and self._th._sem is not None:
            self._th._sem.release()


# -- 故障注入 --


class InjectedFailure(AgentError):
    """注入的 LLM 失败：带 duck-typed response，供错误分类命中。"""

    def __init__(self, status_code: int = 503) -> None:
        super().__init__(f"injected failure ({status_code})")
        self.response = type(
            "_Resp", (), {"status_code": status_code, "headers": {}}
        )()


class FlakyProvider:
    """按概率注入 503 失败，验证重试/熔断行为。seed 固定可复现。"""

    def __init__(
        self, inner: Any, failure_rate: float, seed: int | None = None
    ) -> None:
        self._inner = inner
        self._rate = failure_rate
        self._rng = random.Random(seed)

    async def complete(self, req: Any) -> Any:
        if self._rng.random() < self._rate:
            raise InjectedFailure(503)
        return await self._inner.complete(req)

    def stream(self, req: Any) -> AsyncIterator:
        if self._rng.random() < self._rate:
            raise InjectedFailure(503)
        return self._inner.stream(req)
