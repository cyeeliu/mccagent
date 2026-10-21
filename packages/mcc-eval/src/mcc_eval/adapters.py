"""被测 Agent 适配层：以 TraceEvent 流为契约接入任意 scaffold。

主实现是 MccAdapter——把 mcc 的 AgentLoop 事件流逐条转译为标准轨迹
（TextDelta→llm_chunk、ToolCallStart→tool_call、ToolCallEnd→tool_result）。
MockAdapter 用于无 API key 的链路冒烟与测试。

模型与凭证复用 mcc 的配置约定（环境变量 > settings.json），评测侧
不引入第二份模型配置。Agent 以 headless 方式在独立工作目录跑单任务，
权限模式固定为 bypassPermissions（评测无人值守，无人工确认通道）。
"""

from __future__ import annotations

import asyncio
import random
import tempfile
import uuid as uuid_module
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from mcc_eval.events import (
    LlmChunk,
    LlmRequest,
    LlmUsage,
    RunFinished,
    RunStarted,
    Seq,
    ToolCall,
    ToolResult,
    TraceEvent,
)


class RunRequest(BaseModel):
    """一次评测任务的执行请求。

    work_dir 由 runner 指定：每 case 一个独立目录，Agent 产物落在
    其中，评分器在其上运行官方测试；适配器据此保持无状态。
    """

    run_id: str
    case_id: str
    task: str
    model: str
    max_turns: int = 20
    work_dir: str | None = None


class AgentAdapter(Protocol):
    """适配协议：执行任务并流式产出标准轨迹事件。"""

    def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]: ...


# ─── mcc 适配器 ──────────────────────────────────────────────────────────


class _Translator:
    """AgentEvent → 轨迹事件的转译状态机。

    每轮 LLM 调用的首个产出事件前补一条 llm_request（模型流上没有
    显式的请求边界信号）；UserEcho / Notice / 权限事件不进轨迹。
    """

    def __init__(self, req: RunRequest) -> None:
        self._req = req
        self._seq = Seq()
        self._call = 0  # LLM 调用序号（工具轮 + 文本轮各算一次）
        self._request_sent = False
        self.turns = 0  # 用户回合数（TurnEnd 计数，单任务恒为 1）
        self.cost_usd = 0.0
        self.error_text: str | None = None

    def _llm_request(self) -> list[TraceEvent]:
        if self._request_sent:
            return []
        self._request_sent = True
        self._call += 1
        return [LlmRequest(seq=self._seq.next(), turn=self._call - 1, input_tokens=0)]

    def translate(self, ev: Any) -> list[TraceEvent]:
        from mcc_core.event import (
            Error as ErrorEvent,
        )
        from mcc_core.event import (
            LlmUsage as LlmUsageEvent,
        )
        from mcc_core.event import (
            Notice,
            PermissionRequest,
            TextDelta,
            ToolCallEnd,
            ToolCallStart,
            TurnEnd,
            UserEcho,
        )

        if isinstance(ev, (UserEcho, Notice, PermissionRequest)):
            return []
        if isinstance(ev, ErrorEvent):
            self.error_text = ev.text
            return []
        if isinstance(ev, TextDelta):
            return self._llm_request() + [
                LlmChunk(seq=self._seq.next(), turn=self._call - 1, delta=ev.text)
            ]
        if isinstance(ev, ToolCallStart):
            return self._llm_request() + [
                ToolCall(
                    seq=self._seq.next(),
                    turn=self._call - 1,
                    call_id=ev.id,
                    tool_name=ev.name,
                    # mcc 事件只带参数预览；评分侧的 args_regex 在预览上匹配，
                    # 置 None 会让所有参数断言恒假
                    args=ev.args_preview,
                )
            ]
        if isinstance(ev, ToolCallEnd):
            return [
                ToolResult(
                    seq=self._seq.next(),
                    call_id=ev.id,
                    output=ev.output,
                    is_error=ev.is_error,
                    duration_ms=0,
                )
            ]
        if isinstance(ev, LlmUsageEvent):
            # 每轮流完的用量快照：token 统计的权威来源（llm_request
            # 在请求时机 input_tokens 多半为 0——那时用量还没回来）。
            # 空产出轮（无文本无工具）此前没有事件触发 llm_request，
            # 在此补发——轮次标记与用量一并可见。
            #
            # 同时复位轮内标记：一轮 LLM 调用以此结束，下一轮产出
            # 事件重新触发 llm_request（turn 递增）。工具轮循环中
            # AgentLoop 不发 TurnEnd（其语义是任务结束），若只靠
            # TurnEnd 复位，第二轮起 llm_request 会静默消失
            out = self._llm_request()
            out.append(
                LlmUsage(
                    seq=self._seq.next(),
                    turn=self._call - 1,
                    input_tokens=ev.input_tokens,
                    output_tokens=ev.output_tokens,
                    cost_usd=ev.cost_usd,
                )
            )
            self._request_sent = False
            return out
        if isinstance(ev, TurnEnd):
            # TurnEnd.cost_usd 是会话累计值，取末轮
            self.cost_usd = ev.cost_usd
            self.turns += 1
            self._request_sent = False
            return []
        return []

    def run_started(self) -> RunStarted:
        return RunStarted(
            seq=self._seq.next(),
            run_id=self._req.run_id,
            case_id=self._req.case_id,
            task=self._req.task,
            model=self._req.model,
        )

    def run_finished(self) -> RunFinished:
        # 执行层只报 error / passed 二态：是否有价值交由评分器判定，
        # 有官方测试或断言配置时以评分结果为准。error 时必须带
        # reason（AgentLoop 的异常文本）——LLM 认证失败 / 网关不可达
        # 这类第一步就死的 run，原因要直达轨迹与报告
        status = "error" if self.error_text is not None else "passed"
        return RunFinished(
            seq=self._seq.next(),
            status=status,
            score=None,
            cost_usd=self.cost_usd,
            turns=self.turns,
            reason=self.error_text,
        )


class MccAdapter:
    """以 mcc AgentLoop 为被测对象：headless 跑单任务并转译轨迹。

    llm / registry / work_dir 可注入（测试替身）；缺省惰性构建——
    provider 经 mcc-config 统一构建（settings.json llm 段与
    MCC_LLM_BASE_URL 等环境变量共用一套优先级），工作目录由
    runner 指定以保证任务之间互不污染。
    """

    def __init__(
        self,
        *,
        work_dir: str | Path | None = None,
        llm: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        self._work_dir = Path(work_dir) if work_dir else None
        self._llm = llm
        self._registry = registry

    async def _build_agent(self, req: RunRequest) -> Any:
        """构建被测 AgentLoop；llm 未注入时按 mcc-config 统一约定。

        provider 构建读取当前目录的 mcc settings（含 llm 段的
        base_url / api_key），与 CLI 同一入口同一优先级；配置加载
        失败不阻断评测（环境变量约定仍可用）。
        """
        from mcc_config import PermissionConfig, build_llm_provider, load
        from mcc_core import ToolContext
        from mcc_harness import AgentLoopBuilder, PermissionChecker
        from mcc_tools import default_registry

        llm = self._llm
        if llm is None:
            try:
                config = await load(Path.cwd())
            except Exception:
                config = None
            llm = build_llm_provider(config=config)

        if req.work_dir:
            work_dir = Path(req.work_dir)
        elif self._work_dir:
            work_dir = self._work_dir
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="eval-mcc-"))
        ctx = ToolContext(cwd=work_dir, session_id=req.run_id)
        checker = PermissionChecker(PermissionConfig(mode="bypassPermissions"))
        return (
            AgentLoopBuilder(llm, self._registry or default_registry(), ctx)
            .model(req.model)
            .permission(checker)
            .max_iterations(req.max_turns)
            .build()
        )

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        translator = _Translator(req)
        yield translator.run_started()

        # push（AgentLoop.run_streaming）→ pull（async generator）桥接
        queue: asyncio.Queue = asyncio.Queue()

        async def worker() -> None:
            try:
                agent = await self._build_agent(req)
                await agent.run_streaming(req.task, queue)
            except Exception as exc:
                from mcc_core.event import Error as ErrorEvent

                await queue.put(ErrorEvent(text=str(exc)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(worker())
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                for trace_ev in translator.translate(ev):
                    yield trace_ev
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        yield translator.run_finished()


# ─── Mock 适配器 ─────────────────────────────────────────────────────────


class MockAdapter:
    """模拟三轮编码 Agent（grep 定位 → edit 修改 → bash 验证）。

    不需要 API key，用于链路冒烟与测试。任务文本含 "fail" 时模拟
    失败 run，方便验证失败路径的报告呈现。
    """

    def __init__(self, pace_s: float = 0.0) -> None:
        self.pace_s = pace_s  # 事件间隔；测试里取 0 保证快速完成

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        seq = Seq()
        should_fail = "fail" in req.task.lower()
        yield RunStarted(
            seq=seq.next(),
            run_id=req.run_id,
            case_id=req.case_id,
            task=req.task,
            model=req.model,
        )

        plan = [
            (
                "grep",
                {"pattern": "divide", "path": "src/"},
                "src/math.py:12: def divide(",
                False,
            ),
            ("edit", {"path": "src/math.py"}, "+4 -1 lines", False),
            (
                "bash",
                {"command": "python -m pytest"},
                "2 failed" if should_fail else "12 passed",
                should_fail,
            ),
        ]

        turn = 0
        for tool, args, output, is_error in plan:
            yield LlmRequest(seq=seq.next(), turn=turn, input_tokens=1200 + turn * 800)
            for delta in (f"第 {turn + 1} 步：", f"我将调用 {tool}。"):
                if self.pace_s:
                    await asyncio.sleep(self.pace_s / 2)
                yield LlmChunk(seq=seq.next(), turn=turn, delta=delta)

            call_id = uuid_module.uuid4().hex[:8]
            yield ToolCall(seq=seq.next(), turn=turn, call_id=call_id, tool_name=tool, args=args)
            if self.pace_s:
                await asyncio.sleep(self.pace_s)
            yield ToolResult(
                seq=seq.next(),
                call_id=call_id,
                output=output,
                is_error=is_error,
                duration_ms=random.randint(8, 2200),
            )
            turn += 1

        # 模拟 Agent 的最终汇报（output_contains 断言的判定对象）
        summary = "FAILED" if should_fail else "PASSED"
        yield LlmChunk(seq=seq.next(), turn=turn - 1, delta=f"任务结束：{summary}")

        yield RunFinished(
            seq=seq.next(),
            status="failed" if should_fail else "passed",
            score=0.0 if should_fail else 1.0,
            cost_usd=round(random.uniform(0.01, 0.06), 4),
            turns=turn,
        )
