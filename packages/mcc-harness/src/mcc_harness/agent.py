"""Agent 主循环：驱动 LLM 调用与工具执行的迭代循环。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from mcc_core import (
    AgentError,
    ContentBlock,
    Message,
    PermissionDecision,
    Role,
    TextBlock,
    ToolContext,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from mcc_core.event import (
    AgentEvent,
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_core.event import (
    PermissionRequest as PermissionRequestEvent,
)
from mcc_llm import (
    CompleteRequest,
    LlmProvider,
    MessageStop,
    ToolUseInputDelta,
    ToolUseStart,
)
from mcc_llm import (
    TextDelta as LlmTextDelta,
)
from mcc_tools import ToolRegistry

from .hooks import HookContext, HookRegistry, HookType
from .permission import (
    Allow,
    Ask,
    BashAction,
    Deny,
    PathAction,
    PermissionChecker,
    PermissionRequest,
)


@dataclass
class AgentRun:
    """单次 agent 运行的最终结果。"""

    final_text: str
    iterations: int
    total_usage: Usage
    cost_usd: float


def model_price_per_1m(model: str) -> tuple[float, float]:
    """返回 (input_usd_per_1m, output_usd_per_1m)。"""
    if "claude-opus-4" in model or "opus-4" in model:
        return (15.0, 75.0)
    if "claude-sonnet-4" in model or "sonnet-4" in model:
        return (3.0, 15.0)
    if "claude-haiku-4" in model or "haiku-4" in model:
        return (0.80, 4.0)
    if "gpt-4o" in model:
        return (2.5, 10.0)
    if "gpt-4-turbo" in model:
        return (10.0, 30.0)
    if "gpt-3.5" in model:
        return (0.5, 1.5)
    return (3.0, 15.0)


def price_usd(model: str, usage: Usage) -> float:
    """根据模型定价与 token 用量计算 USD 成本。"""
    in_m, out_m = model_price_per_1m(model)
    return (usage.input_tokens / 1_000_000.0) * in_m + (
        usage.output_tokens / 1_000_000.0
    ) * out_m


def permission_request_for(tool_name: str, input: dict[str, Any]) -> PermissionRequest:
    """将工具调用映射到权限检查请求。

    必须传递真实的 command / path，否则 deny 规则无法匹配，
    权限系统被静默绕过。
    """
    path = str(input.get("path", "."))
    match tool_name:
        case "read_file" | "list_dir" | "grep" | "glob":
            return PermissionRequest(category="Read", action=PathAction(path=path))
        case "write_file":
            return PermissionRequest(category="Write", action=PathAction(path=path))
        case "edit_file":
            return PermissionRequest(category="Edit", action=PathAction(path=path))
        case "run_bash":
            return PermissionRequest(
                category="Bash",
                action=BashAction(cmd=str(input.get("command", ""))),
            )
        case _:
            return PermissionRequest(
                category="Bash", action=BashAction(cmd=tool_name)
            )


class AgentLoop:
    """Agent 主循环，驱动 LLM 调用与工具执行的迭代。"""

    def __init__(
        self,
        llm: LlmProvider,
        registry: ToolRegistry,
        ctx: ToolContext,
        system: str = "",
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        max_iterations: int = 40,
        temperature: float = 0.0,
        permission: PermissionChecker | None = None,
        max_cost_usd: float = 0.0,
        hooks: HookRegistry | None = None,
        permission_responses: asyncio.Queue[PermissionDecision] | None = None,
        permission_timeout_secs: float = 120.0,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.permission = permission
        self.max_cost_usd = max_cost_usd
        self.hooks = hooks
        self.permission_responses = permission_responses
        self.permission_timeout_secs = permission_timeout_secs

    async def run_streaming(
        self, user_input: str, tx: asyncio.Queue[AgentEvent]
    ) -> AgentRun:
        """流式运行，实时发送 AgentEvent 到队列，返回最终统计。"""
        await tx.put(UserEcho(text=user_input))
        messages: list[Message] = [Message.user(user_input)]
        total_usage = Usage()
        total_cost = 0.0

        for iteration in range(1, self.max_iterations + 1):
            if self.max_cost_usd > 0 and total_cost >= self.max_cost_usd:
                await tx.put(
                    Notice(
                        text=f"Budget limit ${self.max_cost_usd:.4} reached "
                        f"after {iteration - 1} iteration(s)"
                    )
                )
                raise AgentError(
                    f"cost ${total_cost:.4} >= limit ${self.max_cost_usd:.4}"
                )

            if self.hooks:
                ctx = HookContext(
                    hook_type=HookType.ON_TURN_START,
                    session_id=self.ctx.session_id,
                    cost_usd=total_cost,
                    iteration=iteration,
                    cwd=self.ctx.cwd,
                )
                for hr in await self.hooks.run_hooks(HookType.ON_TURN_START, ctx):
                    if hr.message:
                        await tx.put(Notice(text=hr.message))

            req = CompleteRequest(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                system=self.system or None,
                temperature=self.temperature,
                tools=self.registry.as_api_schema(),
            )

            blocks, stop_reason, turn_usage = await self._call_with_retry(req, tx)
            turn_cost = price_usd(self.model, turn_usage)
            total_cost += turn_cost
            total_usage = total_usage + turn_usage
            messages.append(Message(role=Role.assistant, content=blocks))

            if stop_reason in ("end_turn", "stop_sequence", None):
                final_text = "\n".join(
                    b.text for b in blocks if isinstance(b, TextBlock)
                )
                if self.hooks:
                    ctx = HookContext(
                        hook_type=HookType.ON_TURN_END,
                        session_id=self.ctx.session_id,
                        cost_usd=total_cost,
                        iteration=iteration,
                        cwd=self.ctx.cwd,
                    )
                    for hr in await self.hooks.run_hooks(HookType.ON_TURN_END, ctx):
                        if hr.message:
                            await tx.put(Notice(text=hr.message))
                await tx.put(TurnEnd(cost_usd=total_cost))
                return AgentRun(final_text, iteration, total_usage, total_cost)

            if stop_reason == "tool_use":
                tool_results = await self._execute_tools(blocks, tx, iteration)
                messages.append(Message(role=Role.tool, content=tool_results))
            else:
                if self.hooks:
                    ctx = HookContext(
                        hook_type=HookType.ON_ERROR,
                        session_id=self.ctx.session_id,
                        cost_usd=total_cost,
                        iteration=iteration,
                        cwd=self.ctx.cwd,
                    )
                    await self.hooks.run_hooks(HookType.ON_ERROR, ctx)
                raise AgentError(f"unexpected stop_reason: {stop_reason}")

        raise AgentError(f"exceeded max_iterations={self.max_iterations}")

    async def run(self, user_input: str) -> AgentRun:
        """非流式批量运行，丢弃事件。"""
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        return await self.run_streaming(user_input, queue)

    async def _call_with_retry(
        self, req: CompleteRequest, tx: asyncio.Queue[AgentEvent]
    ) -> tuple[list[ContentBlock], str | None, Usage]:
        """带指数退避的 LLM 调用重试。"""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                return await self._call_streaming_once(req, tx)
            except Exception as e:
                msg = str(e)
                retryable = any(
                    s in msg for s in ("429", "500", "502", "503", "overloaded")
                )
                if not retryable or attempt >= max_retries:
                    raise
                delay = 0.5 * (2**attempt)
                await tx.put(
                    Notice(
                        text=f"API error (attempt {attempt}/{max_retries}), "
                        f"retrying in {delay:.1f}s..."
                    )
                )
                await asyncio.sleep(delay)
        raise AgentError("retry exhausted")

    async def _call_streaming_once(
        self, req: CompleteRequest, tx: asyncio.Queue[AgentEvent]
    ) -> tuple[list[ContentBlock], str | None, Usage]:
        """单次流式 LLM 调用。

        工具调用块按流式到达顺序保存在列表中（非字典），
        确保 assistant 消息顺序确定且与模型意图一致。
        """
        text_buf: list[str] = []
        tool_slots: list[tuple[str, str, str]] = []
        current_tool_idx: int | None = None
        stop_reason: str | None = None
        usage = Usage()

        async for ev in self.llm.stream(req):
            if isinstance(ev, LlmTextDelta):
                await tx.put(TextDelta(text=ev.text))
                text_buf.append(ev.text)
            elif isinstance(ev, ToolUseStart):
                current_tool_idx = len(tool_slots)
                await tx.put(ToolCallStart(id=ev.id, name=ev.name, args_preview=""))
                tool_slots.append((ev.id, ev.name, ""))
            elif isinstance(ev, ToolUseInputDelta):
                if current_tool_idx is not None:
                    sid, sname, sbuf = tool_slots[current_tool_idx]
                    tool_slots[current_tool_idx] = (
                        sid,
                        sname,
                        sbuf + ev.partial_json,
                    )
            elif isinstance(ev, MessageStop):
                stop_reason = ev.stop_reason
                usage = ev.usage

        blocks: list[ContentBlock] = []
        if text_buf:
            blocks.append(TextBlock(text="".join(text_buf)))
        for tid, tname, json_str in tool_slots:
            try:
                inp = json.loads(json_str) if json_str else {}
            except json.JSONDecodeError:
                inp = {}
            blocks.append(ToolUseBlock(id=tid, name=tname, input=inp))

        return blocks, stop_reason, usage

    async def _execute_tools(
        self,
        blocks: list[ContentBlock],
        tx: asyncio.Queue[AgentEvent],
        iteration: int = 0,
    ) -> list[ContentBlock]:
        """并行执行工具调用。

        每次工具调用前走权限检查与 pre-tool hooks，
        调用后走 post-tool hooks。
        """
        calls = [
            (b.id, b.name, b.input)
            for b in blocks
            if isinstance(b, ToolUseBlock)
        ]

        async def run_one(
            tool_id: str, tool_name: str, tool_input: dict[str, Any]
        ) -> tuple[str, str, bool]:
            decision = (
                self.permission.check(permission_request_for(tool_name, tool_input))
                if self.permission
                else Allow()
            )
            match decision:
                case Deny(reason=reason):
                    return (tool_id, f"permission denied: {reason}", True)
                case Ask(message=message):
                    req_id = abs(hash(tool_id)) % (10**8)
                    await tx.put(
                        PermissionRequestEvent(
                            id=req_id,
                            message=f"{tool_name}: {message}",
                            tool_name=tool_name,
                            tool_input=tool_input,
                        )
                    )
                    user_decision = await self._await_permission_decision(req_id)
                    if user_decision.action == "feedback":
                        return (
                            tool_id,
                            f"permission denied by user: {user_decision.feedback}",
                            True,
                        )
                    if user_decision.action == "deny":
                        return (tool_id, "permission denied by user", True)
                    if user_decision.action == "allow_always" and self.permission:
                        preq = permission_request_for(tool_name, tool_input)
                        self.permission.allow_always(preq.category, preq.action)
                case Allow():
                    pass

            if self.hooks:
                ctx = HookContext(
                    hook_type=HookType.PRE_TOOL,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    session_id=self.ctx.session_id,
                    iteration=iteration,
                    cwd=self.ctx.cwd,
                )
                for hr in await self.hooks.run_hooks(HookType.PRE_TOOL, ctx):
                    if hr.message:
                        await tx.put(Notice(text=hr.message))
                    if not hr.allow:
                        return (tool_id, f"blocked by hook: {hr.message}", True)
                    if hr.modified_input is not None:
                        tool_input = hr.modified_input

            tool = self.registry.get(tool_name)
            if tool is None:
                return (tool_id, f"unknown tool: {tool_name}", True)
            out = await tool.execute(tool_input, self.ctx)
            content = out.content
            is_error = out.is_error

            if self.hooks:
                ctx = HookContext(
                    hook_type=HookType.POST_TOOL,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    tool_output=content,
                    is_error=is_error,
                    session_id=self.ctx.session_id,
                    iteration=iteration,
                    cwd=self.ctx.cwd,
                )
                for hr in await self.hooks.run_hooks(HookType.POST_TOOL, ctx):
                    if hr.message:
                        await tx.put(Notice(text=hr.message))
                    if hr.modified_output is not None:
                        content = hr.modified_output

            return (tool_id, content, is_error)

        tasks = [run_one(tid, tname, tinput) for tid, tname, tinput in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_blocks: list[ContentBlock] = []
        for i, result in enumerate(results):
            tool_id = calls[i][0]
            if isinstance(result, Exception):
                result = (tool_id, f"internal error: {result}", True)
            tid, content, is_error = result
            await tx.put(ToolCallEnd(id=tid, output=content[:500], is_error=is_error))
            result_blocks.append(
                ToolResultBlock(tool_use_id=tid, content=content, is_error=is_error)
            )
        return result_blocks

    async def _await_permission_decision(self, req_id: int) -> PermissionDecision:
        """等待用户权限决策。

        - 无响应通道（headless）或超时：默认拒绝
        - request_id 不匹配的过期决策放回队列，继续等待
        """
        deny = PermissionDecision(request_id=req_id, action="deny")
        if self.permission_responses is None:
            return deny

        while True:
            try:
                candidate = await asyncio.wait_for(
                    self.permission_responses.get(),
                    timeout=self.permission_timeout_secs,
                )
            except TimeoutError:
                return deny
            if candidate.request_id == req_id:
                return candidate
            await self.permission_responses.put(candidate)
            await asyncio.sleep(0)


class AgentLoopBuilder:
    """AgentLoop 构建器。"""

    def __init__(
        self, llm: LlmProvider, registry: ToolRegistry, ctx: ToolContext
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._ctx = ctx
        self._system = ""
        self._model = "claude-sonnet-4-6"
        self._max_tokens = 8192
        self._max_iterations = 40
        self._temperature = 0.0
        self._permission: PermissionChecker | None = None
        self._max_cost_usd = 0.0
        self._hooks: HookRegistry | None = None
        self._permission_responses: asyncio.Queue[PermissionDecision] | None = None
        self._permission_timeout_secs = 120.0

    def system(self, s: str) -> AgentLoopBuilder:
        self._system = s
        return self

    def model(self, m: str) -> AgentLoopBuilder:
        self._model = m
        return self

    def max_tokens(self, n: int) -> AgentLoopBuilder:
        self._max_tokens = n
        return self

    def max_iterations(self, n: int) -> AgentLoopBuilder:
        self._max_iterations = n
        return self

    def temperature(self, t: float) -> AgentLoopBuilder:
        self._temperature = t
        return self

    def permission(self, p: PermissionChecker) -> AgentLoopBuilder:
        self._permission = p
        return self

    def max_cost_usd(self, usd: float) -> AgentLoopBuilder:
        self._max_cost_usd = usd
        return self

    def hooks(self, h: HookRegistry) -> AgentLoopBuilder:
        self._hooks = h
        return self

    def permission_responses(
        self, q: asyncio.Queue[PermissionDecision]
    ) -> AgentLoopBuilder:
        self._permission_responses = q
        return self

    def permission_timeout_secs(self, secs: float) -> AgentLoopBuilder:
        self._permission_timeout_secs = secs
        return self

    def build(self) -> AgentLoop:
        return AgentLoop(
            llm=self._llm,
            registry=self._registry,
            ctx=self._ctx,
            system=self._system,
            model=self._model,
            max_tokens=self._max_tokens,
            max_iterations=self._max_iterations,
            temperature=self._temperature,
            permission=self._permission,
            max_cost_usd=self._max_cost_usd,
            hooks=self._hooks,
            permission_responses=self._permission_responses,
            permission_timeout_secs=self._permission_timeout_secs,
        )
