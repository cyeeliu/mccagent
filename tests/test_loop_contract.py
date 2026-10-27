"""主循环契约测试：拦截器执行序、守卫洋葱序、注入/事件通道、builder 兼容。

把 run_streaming 各注入点的隐式顺序固化为显式契约——拦截器重构
或新增机制时，顺序静默变更会被这些断言拦住。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from mcc_core import Notice, ToolContext, Usage
from mcc_harness import (
    AgentLoopBuilder,
    CircuitBreaker,
    RetryPolicy,
)
from mcc_harness.interceptors import (
    IterationContext,
    TurnTimeoutInterceptor,
)
from mcc_llm import CompleteRequest
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


class RecordingInterceptor:
    """记录执行序列；可选注入一条 user 消息验证注入通道。"""

    def __init__(self, name: str, log: list[str], inject: str | None = None) -> None:
        self.name = name
        self._log = log
        self._inject = inject

    async def on_iteration(self, ictx: IterationContext) -> None:
        self._log.append(f"iter:{self.name}:{ictx.iteration}")
        if self._inject is not None and ictx.iteration == 1:
            from mcc_core import Message

            ictx.messages.append(Message.user(self._inject))
            await ictx.tx.put(Notice(text=f"injected by {self.name}"))


class RecordingGuard:
    """记录守卫进出序（洋葱模型断言用）。"""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    async def wrap(self, loop, req: CompleteRequest, tx, invoke):
        self._log.append(f"guard-in:{self.name}")
        result = await invoke(req, tx)
        self._log.append(f"guard-out:{self.name}")
        return result


def _make_agent(ctx: ToolContext, llm=None, **kw):
    builder = (
        AgentLoopBuilder(llm or MockLlm([single_text_turn("done")]), ToolRegistry(), ctx)
        .model("claude-haiku-4-5-20251001")
    )
    return builder, llm


class TestIterationContract:
    async def test_builtin_order(self, ctx: ToolContext) -> None:
        """内置拦截器顺序：timeout → observability → background → cron →
        budget → hooks（有各注入时）。"""
        from mcc_harness import CronScheduler
        from mcc_harness.background import BackgroundTaskManager
        from mcc_tools.sandbox import DirectExecutor

        background = BackgroundTaskManager(DirectExecutor(), default_timeout=1)
        cron = CronScheduler()

        class _Sink:
            def __init__(self) -> None:
                self.iterations: list[int] = []

            def iteration(self, **kw: Any) -> None:
                self.iterations.append(kw["iteration"])

        sink = _Sink()
        builder, _ = _make_agent(ctx)
        agent = (
            builder.turn_timeout(600)
            .observability(sink)
            .background(background)
            .cron(cron)
            .build()
        )
        names = [it.name for it in agent._interceptors]
        assert names == [
            "turn_timeout", "observability", "background", "cron", "budget"
        ]

    async def test_registration_order_is_execution_order(
        self, ctx: ToolContext
    ) -> None:
        log: list[str] = []
        builder, _ = _make_agent(ctx)
        agent = (
            builder.interceptor(RecordingInterceptor("A", log))
            .interceptor(RecordingInterceptor("B", log))
            .build()
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert log.index("iter:A:1") < log.index("iter:B:1")

    async def test_injected_message_reaches_llm(self, ctx: ToolContext) -> None:
        """拦截器注入的 user 消息出现在 LLM 请求中（注入通道契约）。"""
        captured: list[list] = []

        class CaptureLlm:
            async def complete(self, req: CompleteRequest) -> Any:
                raise AssertionError("not used")

            def stream(self, req: CompleteRequest):
                captured.append(list(req.messages))

                async def _gen():
                    from mcc_llm import MessageStop, TextDelta

                    yield TextDelta(text="ok")
                    yield MessageStop(
                        stop_reason="end_turn",
                        usage=Usage(input_tokens=1, output_tokens=1),
                    )

                return _gen()

        log: list[str] = []
        builder = AgentLoopBuilder(CaptureLlm(), ToolRegistry(), ctx).model(
            "claude-haiku-4-5-20251001"
        )
        agent = builder.interceptor(
            RecordingInterceptor("inj", log, inject="injected-content")
        ).build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert captured
        texts = [
            getattr(b, "text", "")
            for m in captured[0]
            for b in m.content
        ]
        assert "injected-content" in texts

    async def test_notice_reaches_event_stream(self, ctx: ToolContext) -> None:
        log: list[str] = []
        builder, _ = _make_agent(ctx)
        agent = builder.interceptor(
            RecordingInterceptor("inj", log, inject="x")
        ).build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        notices = []
        while not queue.empty():
            ev = queue.get_nowait()
            if isinstance(ev, Notice):
                notices.append(ev.text)
        assert "injected by inj" in notices


class TestGuardContract:
    async def test_guard_onion_order(self, ctx: ToolContext) -> None:
        """洋葱序：注册在前的守卫在外层。"""
        log: list[str] = []
        builder, _ = _make_agent(ctx)
        agent = (
            builder.guard(RecordingGuard("outer", log))
            .guard(RecordingGuard("inner", log))
            .build()
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert log == [
            "guard-in:outer",
            "guard-in:inner",
            "guard-out:inner",
            "guard-out:outer",
        ]

    async def test_reliability_guard_registered_by_default(
        self, ctx: ToolContext
    ) -> None:
        builder, _ = _make_agent(ctx)
        agent = builder.build()
        assert [g.name for g in agent._guards] == ["reliability"]

    async def test_custom_guard_wraps_reliability(self, ctx: ToolContext) -> None:
        """自定义守卫在可靠性守卫内层（reliability 最先注册=最外）。"""
        log: list[str] = []
        builder, _ = _make_agent(ctx)
        agent = (
            builder.retry_policy(RetryPolicy(max_attempts=2, jitter=False))
            .guard(RecordingGuard("custom", log))
            .build()
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert log[0] == "guard-in:custom"


class TestBuilderCompatibility:
    async def test_old_builder_methods_map_to_interceptors(
        self, ctx: ToolContext
    ) -> None:
        """旧 builder 方法注册的拦截器名与内置顺序一致。"""
        from mcc_harness.background import BackgroundTaskManager
        from mcc_tools.sandbox import DirectExecutor

        bg = BackgroundTaskManager(DirectExecutor(), default_timeout=1)

        agent = _make_agent(ctx)[0].turn_timeout(600).background(bg).build()
        names = [it.name for it in agent._interceptors]
        assert names == ["turn_timeout", "background", "budget"]

    async def test_custom_interceptors_run_after_builtins(
        self, ctx: ToolContext
    ) -> None:
        """协议注册的自定义拦截器在内置段之后执行。"""
        from mcc_harness.background import BackgroundTaskManager
        from mcc_tools.sandbox import DirectExecutor

        bg = BackgroundTaskManager(DirectExecutor(), default_timeout=1)
        log: list[str] = []
        agent = (
            _make_agent(ctx)[0]
            .background(bg)
            .interceptor(RecordingInterceptor("custom", log))
            .build()
        )
        names = [it.name for it in agent._interceptors]
        assert names.index("background") < names.index("custom")
        assert names.index("budget") < names.index("custom")

    async def test_behavior_unchanged_with_retry_defaults(
        self, ctx: ToolContext
    ) -> None:
        """无注入时行为回归：默认重试守卫生效，一轮即返回。"""
        builder, _ = _make_agent(ctx)
        agent = builder.build()
        queue: asyncio.Queue = asyncio.Queue()
        run = await agent.run_streaming("hi", queue)
        assert run.final_text == "done"
        assert run.iterations == 1
