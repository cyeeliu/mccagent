"""AgentLoop 扩展协议：迭代拦截器、调用守卫与迭代上下文。

主循环的可扩展注入面收敛为两个协议：

- ``IterationInterceptor``：每轮迭代首按注册序执行（超时检查 /
  观测埋点 / 后台与 cron 注入 / 预算 / 外部 hooks），异常即中断
  本轮（预算超限等语义保留）；
- ``LlmCallGuard``：洋葱模型包裹一次 LLM 调用（重试/熔断）。

``IterationContext`` 是拦截器与主循环的交互面：``messages`` 可变
（注入用户消息），``tx`` 可发事件。builder 的旧注入方法保留为
薄包装，内部转为本模块 adapter；新机制直接实现协议即可，不再
扩展 AgentLoop.__init__ 签名。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mcc_core import AgentError, AgentEvent, Message, Notice
from mcc_core.types import ContentBlock, Usage
from mcc_llm import CompleteRequest

if TYPE_CHECKING:
    from mcc_harness.agent import AgentLoop


@dataclass
class IterationContext:
    """一轮迭代首的上下文：拦截器与主循环的全部交互面。"""

    loop: AgentLoop
    messages: list[Message]
    tx: asyncio.Queue[AgentEvent]
    iteration: int
    total_cost: float


CallFn = Callable[
    [CompleteRequest, "asyncio.Queue[AgentEvent]"],
    Awaitable["tuple[list[ContentBlock], str | None, Usage]"],
]


class IterationInterceptor(Protocol):
    """迭代期拦截器：每轮迭代首按注册序执行。"""

    name: str

    async def on_iteration(self, ictx: IterationContext) -> None: ...


class LlmCallGuard(Protocol):
    """LLM 调用守卫：洋葱模型，``invoke`` 是内层调用。"""

    name: str

    async def wrap(
        self,
        loop: AgentLoop,
        req: CompleteRequest,
        tx: asyncio.Queue[AgentEvent],
        invoke: CallFn,
    ) -> tuple[list[ContentBlock], str | None, Usage]: ...


# ---------- 内置 adapter（迁移自 AgentLoop 内联逻辑） ----------


class TurnTimeoutInterceptor:
    """turn 级超时：协作式迭代首检查（不取消进行中的调用）。"""

    name = "turn_timeout"

    def __init__(self, timeout_secs: float) -> None:
        self._timeout = timeout_secs
        self._started_at: float | None = None

    async def on_iteration(self, ictx: IterationContext) -> None:
        if ictx.iteration == 1:
            self._started_at = time.monotonic()
        assert self._started_at is not None
        if time.monotonic() - self._started_at >= self._timeout:
            await ictx.tx.put(
                Notice(
                    text=f"turn timed out after {self._timeout:.0f}s "
                    f"({ictx.iteration - 1} iteration(s) completed)"
                )
            )
            raise AgentError(f"turn timed out after {self._timeout:.0f}s")


class ObservabilityInterceptor:
    """观测埋点：每轮迭代计数。"""

    name = "observability"

    def __init__(self, sink: Any) -> None:
        self._sink = sink

    async def on_iteration(self, ictx: IterationContext) -> None:
        self._sink.iteration(iteration=ictx.iteration)


class BackgroundInterceptor:
    """后台任务结果注入：collect → user 消息 + Notice。"""

    name = "background"

    def __init__(self, manager: Any) -> None:
        self._manager = manager

    async def on_iteration(self, ictx: IterationContext) -> None:
        from mcc_harness.background import format_background_results

        results = await self._manager.collect()
        if results:
            notification = format_background_results(results)
            ictx.messages.append(Message.user(notification))
            await ictx.tx.put(Notice(text=notification))


class CronInterceptor:
    """Cron 定时任务注入：consume → user 消息 + Notice + acknowledge。"""

    name = "cron"

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    async def on_iteration(self, ictx: IterationContext) -> None:
        from mcc_harness.cron import format_cron_delivery

        jobs = await self._scheduler.consume()
        if jobs:
            notification = format_cron_delivery(jobs)
            ictx.messages.append(Message.user(notification))
            await ictx.tx.put(Notice(text=notification))
            await self._scheduler.acknowledge(jobs)


class BudgetInterceptor:
    """预算护栏：达到阈值发一次警告，达到上限中断。"""

    name = "budget"

    def __init__(self) -> None:
        self._warned = False

    async def on_iteration(self, ictx: IterationContext) -> None:
        if ictx.iteration == 1:
            self._warned = False
        loop = ictx.loop
        limit = loop.max_cost_usd
        if limit <= 0:
            return
        total = ictx.total_cost
        if not self._warned and total >= limit * loop.warn_threshold:
            self._warned = True
            await ictx.tx.put(
                Notice(
                    text=f"Budget warning: ${total:.4} / ${limit:.4} "
                    f"({total / limit:.0%})"
                )
            )
        if total >= limit:
            await ictx.tx.put(
                Notice(
                    text=f"Budget limit ${limit:.4} reached "
                    f"after {ictx.iteration - 1} iteration(s)"
                )
            )
            raise AgentError(f"cost ${total:.4} >= limit ${limit:.4}")


class HooksInterceptor:
    """外部 hooks：每轮迭代首运行 ON_TURN_START 钩子。"""

    name = "hooks"

    def __init__(self, hooks: Any) -> None:
        self._hooks = hooks

    async def on_iteration(self, ictx: IterationContext) -> None:
        from mcc_harness.hooks import HookContext, HookType

        ctx = HookContext(
            hook_type=HookType.ON_TURN_START,
            session_id=ictx.loop.ctx.session_id,
            cost_usd=ictx.total_cost,
            iteration=ictx.iteration,
            depth=ictx.loop.ctx.depth,
            cwd=ictx.loop.ctx.cwd,
        )
        for hr in await self._hooks.run_hooks(HookType.ON_TURN_START, ctx):
            if hr.message:
                await ictx.tx.put(Notice(text=hr.message))


class ReliabilityGuard:
    """LLM 调用守卫：熔断前置 + 结构化分类重试 + 成功埋点。

    retry_policy 与 circuit_breaker 是同一可靠性领域的两个开关，
    合并为一个守卫（重试判定依赖熔断计数，无法拆成独立洋葱层
    而不互相穿透）。
    """

    name = "reliability"

    def __init__(
        self,
        retry_policy: Any | None = None,
        circuit_breaker: Any | None = None,
    ) -> None:
        self._policy = retry_policy
        self._breaker = circuit_breaker

    async def wrap(
        self,
        loop: AgentLoop,
        req: CompleteRequest,
        tx: asyncio.Queue[AgentEvent],
        invoke: CallFn,
    ) -> tuple[list[ContentBlock], str | None, Usage]:
        from mcc_harness.reliability import RetryPolicy, classify_llm_error

        policy = self._policy or RetryPolicy()
        if self._breaker is not None:
            try:
                self._breaker.check()
            except Exception:
                if loop.observability is not None:
                    loop.observability.event_counter("agent_circuit_open_total")
                raise
        t0: int | None = None
        if loop.observability is not None:
            from mcc_observability import now_ms

            t0 = now_ms()
        rng = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                blocks, stop_reason, usage = await invoke(req, tx)
            except Exception as e:
                decision = classify_llm_error(e)
                if self._breaker is not None and decision.retryable:
                    self._breaker.record_failure()
                if not decision.retryable or attempt >= policy.max_attempts:
                    raise
                if decision.retry_after_ms is not None:
                    delay = decision.retry_after_ms / 1000.0
                else:
                    if rng is None:
                        import random

                        rng = random.Random()
                    delay = policy.delay(attempt, rng)
                await tx.put(
                    Notice(
                        text=f"{decision.reason} (attempt {attempt}/"
                        f"{policy.max_attempts}), retrying in {delay:.1f}s..."
                    )
                )
                await asyncio.sleep(delay)
                continue
            if self._breaker is not None:
                self._breaker.record_success()
            if loop.observability is not None and t0 is not None:
                from mcc_observability import now_ms as _now_ms

                loop.observability.llm_call(
                    model=loop.model,
                    latency_ms=_now_ms() - t0,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_input_tokens,
                    cache_write_tokens=usage.cache_creation_input_tokens,
                    stop_reason=stop_reason,
                    attempt=attempt,
                )
            return blocks, stop_reason, usage
        raise AgentError("retry exhausted")
