"""Mock 适配器：不需要 API key 就能让整套平台跑起来（默认启用）。

模拟一个三轮的编码 Agent：grep 定位 → edit 修改 → bash 跑测试。
任务文本里含 "fail" 时模拟失败 run，方便演示对比报告与失败过滤。
"""

from __future__ import annotations

import asyncio
import random
import uuid
from collections.abc import AsyncIterator

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


class MockAdapter:
    def __init__(self, pace_s: float = 0.4) -> None:
        self.pace_s = pace_s  # 事件间隔：演示实时流的"节奏感"

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        seq = Seq()
        should_fail = "fail" in req.task.lower()
        yield RunStarted(
            seq=seq.next(), run_id=req.run_id, case_id=req.case_id, task=req.task, model=req.model
        )

        plan = [
            (
                "grep",
                {"pattern": "divide", "path": "src/"},
                "src/math.rs:12: pub fn divide(",
                False,
            ),
            ("edit", {"path": "src/math.rs"}, "+4 -1 lines", False),
            (
                "bash",
                {"command": "cargo test"},
                "test result: FAILED. 1 failed" if should_fail else "test result: ok. 12 passed",
                should_fail,
            ),
        ]

        turn = 0
        for tool, args, output, is_error in plan:
            yield LlmRequest(seq=seq.next(), turn=turn, input_tokens=1200 + turn * 800)
            for delta in (f"第 {turn + 1} 步：", f"我将调用 {tool}。"):
                await asyncio.sleep(self.pace_s / 2)
                yield LlmChunk(seq=seq.next(), turn=turn, delta=delta)

            call_id = uuid.uuid4().hex[:8]
            yield ToolCall(seq=seq.next(), turn=turn, call_id=call_id, tool_name=tool, args=args)
            await asyncio.sleep(self.pace_s)
            yield ToolResult(
                seq=seq.next(),
                call_id=call_id,
                output=output,
                is_error=is_error,
                duration_ms=random.randint(8, 2200),
            )
            turn += 1

        yield RunFinished(
            seq=seq.next(),
            status="failed" if should_fail else "passed",
            score=0.0 if should_fail else 1.0,
            cost_usd=round(random.uniform(0.01, 0.06), 4),
            turns=turn,
        )
