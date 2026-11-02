"""事件源抽象与实现。

EventSource 协议定义 Agent 事件流的统一接口，
前端（TUI/headless）只消费事件，不直接调用 Agent。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, runtime_checkable

from mcc_core import (
    AgentEvent,
    ModelChanged,
    Notice,
    SelectionDecision,
    SelectionOption,
    SelectionRequest,
    SessionChanged,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_harness.goal import GoalController


@runtime_checkable
class EventSource(Protocol):
    """事件源协议：接收用户输入，产出 AgentEvent 事件流。"""

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]: ...


def history_events(history: list, cost_usd: float = 0.0) -> list:
    """把恢复的消息历史转换为 TUI 重放事件序列。

    历史只传给 AgentLoop 时 LLM 具备上下文，但渲染层从未
    收到事件，界面会是空白；此处按 user/assistant/tool 消息
    逐条还原轨迹，末尾以 TurnEnd 收尾（携带会话累计成本）。
    """
    from mcc_core import Role, TextBlock, ToolResultBlock, ToolUseBlock

    events = []
    for m in history:
        if m.role == Role.user:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(UserEcho(text=b.text))
        elif m.role == Role.assistant:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(TextDelta(text=b.text))
                elif isinstance(b, ToolUseBlock):
                    events.append(ToolCallStart(
                        id=b.id, name=b.name,
                        args_preview=json.dumps(b.input, ensure_ascii=False)
                        if b.input else "",
                    ))
        elif m.role == Role.tool:
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    events.append(
                        ToolCallEnd(
                            id=b.tool_use_id,
                            output=b.content[:500],
                            is_error=b.is_error,
                        )
                    )
    events.append(TurnEnd(cost_usd=cost_usd))
    return events


class EchoEventSource:
    """模拟事件源：回显用户输入并模拟流式文本输出。

    用于在前端接入真实 Agent 前演示交互与流式渲染效果。
    """

    def __init__(self, chunk_size: int = 12, delay: float = 0.02) -> None:
        self._chunk_size = chunk_size
        self._delay = delay

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        yield UserEcho(text=prompt)
        yield ToolCallStart(id="c0", name="echo", args_preview=prompt[:60])
        await asyncio.sleep(self._delay)
        yield ToolCallEnd(id="c0", output=f"received: {prompt}", is_error=False)
        yield Notice(text="generating response…")

        response = (
            f"已收到你的请求：「{prompt}」\n"
            "这是一个模拟响应，真实 Agent 集成后将由 LLM 生成。"
        )
        for i in range(0, len(response), self._chunk_size):
            yield TextDelta(text=response[i : i + self._chunk_size])
            await asyncio.sleep(self._delay)

        yield TurnEnd(cost_usd=0.0)


class AgentEventSource(EventSource):
    """将 AgentLoop 适配为 EventSource。

    通过 asyncio.Queue 桥接 AgentLoop.run_streaming 的 push 模型
    与 EventSource.run 的 pull 模型（async generator）。
    """

    def __init__(
        self,
        agent: object,
        slash_resolver: Callable[[str], str | None] | None = None,
        goal_controller: GoalController | None = None,
        team_manager: Any | None = None,
        cron_scheduler: Any | None = None,
        workflow_tool: Any | None = None,
        observability_sink: Any | None = None,
        slash_registry: Any | None = None,
        config: Any | None = None,
        work_dir: Any | None = None,
        skills_catalog: str | None = None,
        skill_items: list[tuple[str, str]] | None = None,
        session_short_id: str | None = None,
        selection_responses: asyncio.Queue[SelectionDecision] | None = None,
    ) -> None:
        self.agent = agent
        self._slash_resolver = slash_resolver
        self._goal_controller = goal_controller
        self.team_manager = team_manager
        self.cron_scheduler = cron_scheduler
        self.workflow_tool = workflow_tool
        self.observability_sink = observability_sink
        self.slash_registry = slash_registry
        self.config = config
        self.work_dir = work_dir
        self.skills_catalog = skills_catalog
        self.skill_items = skill_items or []
        self.session_short_id = session_short_id
        self.selection_responses = selection_responses

    @property
    def goal_controller(self) -> GoalController | None:
        return self._goal_controller

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        # slash 本地命令：/model /config /help /status /goal ——注册表
        # 分发，不进 AgentLoop；prompt 非空时替换输入继续（goal set）。
        # /resume 等命令返回 select 候选时，发 SelectionRequest 等用户
        # 选择后以 /<command> <value> 重新分发（while 循环）。
        if self.slash_registry is not None:
            while True:
                result = await self.slash_registry.dispatch(prompt, self)
                if result is None:
                    break
                if result.model_update:
                    yield ModelChanged(model_id=result.model_update)
                if result.session_update:
                    self.session_short_id = result.session_update
                    yield SessionChanged(session_id=result.session_update)
                if result.text:
                    yield Notice(text=result.text)
                if result.replay:
                    for ev in result.replay:
                        yield ev
                if result.select is not None:
                    if self.selection_responses is None:
                        # headless 回退：文本列出候选后结束
                        lines = [result.select_title or "选择:"]
                        for label, _value in result.select:
                            lines.append(f"  {label}")
                        yield Notice(text="\n".join(lines))
                        yield TurnEnd(cost_usd=0.0)
                        return
                    from uuid import uuid4

                    req_id = f"sel-{uuid4().hex[:8]}"
                    options = [
                        SelectionOption(label=lbl, value=val)
                        for lbl, val in result.select
                    ]
                    yield SelectionRequest(
                        id=req_id,
                        title=result.select_title,
                        options=options,
                    )
                    decision = await self.selection_responses.get()
                    if decision.value is None:
                        yield TurnEnd(cost_usd=0.0)
                        return
                    prompt = f"/{result.select_command} {decision.value}"
                    continue
                if result.prompt is None:
                    yield TurnEnd(cost_usd=0.0)
                    return
                prompt = result.prompt
                break
        if self._slash_resolver is not None:
            resolved = self._slash_resolver(prompt)
            if resolved is not None:
                prompt = resolved
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def worker() -> None:
            try:
                await self.agent.run_streaming(prompt, queue)
            except Exception as exc:
                from mcc_core import Error as ErrorEvent

                await queue.put(ErrorEvent(text=str(exc)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield ev
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
