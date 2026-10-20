"""mini-claude-code 适配器：把 mcc 的 AgentLoop 接入评测平台。

mcc 的事件流与平台轨迹事件几乎一一对应（TextDelta→llm_chunk、
ToolCallStart→tool_call、ToolCallEnd→tool_result），在事件流层转译即可
拿到完整轨迹。Agent 以 headless 方式在独立工作目录跑单任务，
权限模式固定为 bypassPermissions（评测沙箱内无人工确认通道）。

依赖 mcc 系列包（不在 extras 中声明，需手动安装）::

    uv pip install -e ../../packages/mcc-core -e ../../packages/mcc-config \\
        -e ../../packages/mcc-llm -e ../../packages/mcc-tools \\
        -e ../../packages/mcc-session -e ../../packages/mcc-harness
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from runner.adapters.base import RunRequest
from runner.events import (
    LlmChunk,
    LlmRequest,
    RunFinished,
    RunStarted,
    Seq,
    ToolCall,
    ToolResult,
    TraceEvent,
)


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
                    args=None,
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
        # 执行状态只区分 error / passed：是否有价值交由 expectations 评分
        # 覆盖（main.py 的 execute 在有 expectations 时用评分结果替换）
        status = "error" if self.error_text is not None else "passed"
        return RunFinished(
            seq=self._seq.next(),
            status=status,
            score=None,
            cost_usd=self.cost_usd,
            turns=self.turns,
        )


class MccAdapter:
    """以 mcc AgentLoop 为被测对象：headless 跑单任务并转译轨迹。

    llm / registry / work_dir 可注入（测试替身）；缺省惰性构建——
    LLM 走环境变量自动探测，工作目录用每 run 独立的临时目录，
    保证任务之间互不污染。
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

    def _build_agent(self, req: RunRequest) -> Any:
        from mcc_config import PermissionConfig
        from mcc_core import ToolContext
        from mcc_harness import AgentLoopBuilder, PermissionChecker
        from mcc_tools import default_registry

        if self._llm is not None:
            llm = self._llm
        else:
            from mcc_llm import auto_provider_from_env

            llm = auto_provider_from_env()

        work_dir = self._work_dir or Path(tempfile.mkdtemp(prefix="eval-mcc-"))
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
                await self._build_agent(req).run_streaming(req.task, queue)
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
