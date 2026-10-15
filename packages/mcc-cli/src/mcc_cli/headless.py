"""Headless 模式：单次执行 prompt 并输出结果。

stdout 输出最终答案（可管道），stderr 输出工具进度与统计。
"""

from __future__ import annotations

import sys

from mcc_core import (
    Error as ErrorEvent,
)
from mcc_core import (
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
)

from mcc_cli.events import EventSource


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def run_headless(source: EventSource, prompt: str, quiet: bool) -> int:
    """运行 headless 模式，返回退出码（0=成功，1=错误）。"""
    final_text = ""
    cost_usd = 0.0
    had_error = False

    async for ev in source.run(prompt):
        if isinstance(ev, ToolCallStart):
            if not quiet:
                _eprint(f"  ⚙  {ev.name}…")
        elif isinstance(ev, ToolCallEnd):
            if ev.is_error and not quiet:
                _eprint(f"  ✗  tool error: {ev.output[:120]}")
        elif isinstance(ev, Notice):
            if not quiet:
                _eprint(f"  ℹ  {ev.text}")
        elif isinstance(ev, ErrorEvent):
            _eprint(f"  ✗  {ev.text}")
            had_error = True
        elif isinstance(ev, TextDelta):
            final_text += ev.text
        elif isinstance(ev, TurnEnd):
            cost_usd = ev.cost_usd

    if had_error:
        return 1

    print(final_text, flush=True)
    if not quiet:
        _eprint(f"\n[${cost_usd:.5f} USD]")
    return 0
