"""真实 Agent 适配器：Anthropic tool-use loop。

需要 ANTHROPIC_API_KEY 与 ``uv sync --extra anthropic``。
工具刻意只给一个安全的 bash（在临时目录里跑，超时硬限制）——
生产部署应将工具执行收归沙箱服务，此处保持单机可运行。
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
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

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the workspace; returns stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
]

# 简化成本模型（演示用；生产应从 API usage 计费表计算）
PRICE_IN, PRICE_OUT = 3e-6, 15e-6

OUTPUT_LIMIT = 10_000
BASH_TIMEOUT_S = 60


class AnthropicAdapter:
    def __init__(self) -> None:
        from anthropic import AsyncAnthropic  # 惰性导入：不装 extra 也能用 mock

        self.client = AsyncAnthropic()

    async def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]:
        seq = Seq()
        yield RunStarted(
            seq=seq.next(), run_id=req.run_id, case_id=req.case_id, task=req.task, model=req.model
        )

        with tempfile.TemporaryDirectory() as workspace:
            messages: list[dict] = [{"role": "user", "content": req.task}]
            in_tok = out_tok = 0

            for turn in range(req.max_turns):
                yield LlmRequest(seq=seq.next(), turn=turn, input_tokens=in_tok)
                resp = await self.client.messages.create(
                    model=req.model,
                    max_tokens=4096,
                    system="You are a coding agent. Use bash to accomplish the task, "
                    "then summarize what you did.",
                    tools=TOOLS,
                    messages=messages,
                )
                in_tok += resp.usage.input_tokens
                out_tok += resp.usage.output_tokens
                messages.append({"role": "assistant", "content": resp.content})

                for block in resp.content:
                    if block.type == "text" and block.text:
                        yield LlmChunk(seq=seq.next(), turn=turn, delta=block.text)

                if resp.stop_reason != "tool_use":
                    yield RunFinished(
                        seq=seq.next(),
                        status="passed",
                        score=None,
                        cost_usd=in_tok * PRICE_IN + out_tok * PRICE_OUT,
                        turns=turn + 1,
                    )
                    return

                results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    call_id = uuid.uuid4().hex[:8]
                    yield ToolCall(
                        seq=seq.next(),
                        turn=turn,
                        call_id=call_id,
                        tool_name=block.name,
                        args=block.input,
                    )
                    started = time.monotonic()
                    output, is_error = await _run_bash(
                        str(block.input.get("command", "")), cwd=workspace
                    )
                    yield ToolResult(
                        seq=seq.next(),
                        call_id=call_id,
                        output=output[:OUTPUT_LIMIT],
                        is_error=is_error,
                        duration_ms=int((time.monotonic() - started) * 1000),
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output[:OUTPUT_LIMIT],
                        }
                    )
                messages.append({"role": "user", "content": results})

            yield RunFinished(
                seq=seq.next(),
                status="timeout",
                cost_usd=in_tok * PRICE_IN + out_tok * PRICE_OUT,
                turns=req.max_turns,
            )


async def _run_bash(
    command: str, cwd: str, timeout_s: int = BASH_TIMEOUT_S
) -> tuple[str, bool]:
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return out.decode(errors="replace"), proc.returncode != 0
    except TimeoutError:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        return f"timeout after {timeout_s}s", True
