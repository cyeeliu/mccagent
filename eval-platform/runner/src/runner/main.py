"""Runner 主循环：lease → 执行 adapter → 批量上报事件 → complete。

要点：
- Bearer token 鉴权（RUNNER_SECRET），未配置时为开发模式
- 心跳续租后台任务 + 逐事件检查租约存活
- 事件缓冲按 FLUSH_INTERVAL_S 批量上报（JSONL）
- 集成评分模块：有 expectations 时计算 score / status 并覆盖执行状态
- 单 run 失败隔离：异常路径上报 error 后继续领下一个任务
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import time

import httpx

from runner.adapters.base import AgentAdapter, RunRequest
from runner.adapters.mock import MockAdapter
from runner.events import LlmChunk, RunFinished, ToolCall, ToolResult
from runner.scoring import TraceSnapshot
from runner.scoring import score as compute_score

SERVER = os.environ.get("EVAL_SERVER_URL", "http://localhost:8081")
RUNNER_ID = os.environ.get("RUNNER_ID", f"{socket.gethostname()}-{os.getpid()}")
RUNNER_SECRET = os.environ.get("RUNNER_SECRET")  # None → 开发模式（服务端放行）
POLL_IDLE_S = float(os.environ.get("POLL_IDLE_S", "2.0"))
FLUSH_INTERVAL_S = 0.5
HEARTBEAT_INTERVAL_S = 30.0
ERROR_MESSAGE_LIMIT = 1000


def _auth_headers() -> dict[str, str]:
    headers = {"X-Runner-Id": RUNNER_ID}
    if RUNNER_SECRET:
        headers["Authorization"] = f"Bearer {RUNNER_SECRET}"
    return headers


def make_adapter(scaffold: str) -> AgentAdapter:
    """按 profile 的 scaffold 字段构造适配器。"""
    if scaffold == "mock":
        return MockAdapter()
    if scaffold == "anthropic":
        from runner.adapters.anthropic_agent import AnthropicAdapter

        return AnthropicAdapter()
    if scaffold == "langgraph":
        from runner.adapters.langgraph_agent import LangGraphAdapter

        return LangGraphAdapter()
    if scaffold == "mini-claude-code":
        # mcc 包不在 PyPI，须在 runner 环境手动安装（见 adapters/mcc_agent.py）
        from runner.adapters.mcc_agent import MccAdapter

        return MccAdapter()
    raise ValueError(f"unknown scaffold: {scaffold}")


async def heartbeat_loop(http: httpx.AsyncClient, run_id: str) -> None:
    """后台心跳：租约被 reaper 回收时主动失败。"""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        r = await http.post(
            f"/internal/runs/{run_id}/heartbeat",
            json={"runner_id": RUNNER_ID},
        )
        r.raise_for_status()
        if not r.json().get("alive"):
            # 租约被回收（多半是本进程卡顿太久）
            raise RuntimeError("lease lost")


async def execute(http: httpx.AsyncClient, lease: dict) -> None:
    """执行单个 run：跑 adapter、攒事件批量上报、评分、complete。"""
    run_id = lease["run_id"]
    expectations: list[dict] = lease.get("expectations") or []

    buf: list[str] = []
    snap = TraceSnapshot()
    final: RunFinished | None = None
    t_start = time.monotonic()

    async def flush() -> None:
        if buf:
            lines, buf[:] = "\n".join(buf), []
            r = await http.post(f"/internal/runs/{run_id}/events", content=lines)
            r.raise_for_status()

    def ensure_heartbeat_alive() -> None:
        if hb.done():
            hb.result()

    hb = asyncio.create_task(heartbeat_loop(http, run_id))
    try:
        req = RunRequest(
            run_id=run_id,
            case_id=lease["case_id"],
            task=lease["task"],
            model=lease["model"],
        )
        adapter = make_adapter(lease["scaffold"])
        last_flush = asyncio.get_running_loop().time()
        async for ev in adapter.run(req):
            ensure_heartbeat_alive()
            # 构建 TraceSnapshot（供评分用）
            if isinstance(ev, ToolCall):
                snap.tool_calls.append(ev)
            elif isinstance(ev, ToolResult):
                snap.tool_results.append(ev)
                if ev.is_error:
                    snap.had_error = True
            elif isinstance(ev, LlmChunk):
                snap.text_outputs.append(ev.delta)
            elif isinstance(ev, RunFinished):
                final = ev

            buf.append(ev.model_dump_json())
            now = asyncio.get_running_loop().time()
            if now - last_flush >= FLUSH_INTERVAL_S:
                await flush()
                ensure_heartbeat_alive()
                last_flush = now

        await flush()
        ensure_heartbeat_alive()
        duration_s = time.monotonic() - t_start

        # 评分：adapter 返回的 RunFinished.status 是执行状态；
        # 如果有 expectations，用评分结果覆盖 status 和 score。
        if final and final.status in ("passed", "failed") and expectations:
            result = compute_score(snap, expectations)
            status = result.status
            score_val = result.score
        elif final:
            status = final.status
            score_val = final.score
        else:
            status = "error"
            score_val = None

        r = await http.post(
            f"/internal/runs/{run_id}/complete",
            json={
                "runner_id": RUNNER_ID,
                "status": status,
                "score": score_val,
                "cost_usd": final.cost_usd if final else None,
                "turns": final.turns if final else None,
                "duration_s": round(duration_s, 3),
                "error": None,
            },
        )
        r.raise_for_status()
        print(
            f"[{RUNNER_ID}] run {run_id} -> {status}"
            + (f" score={score_val:.3f}" if score_val is not None else "")
        )

    except Exception as e:
        # 单 run 失败隔离：上报 error 后继续下一个任务
        duration_s = time.monotonic() - t_start
        print(f"[{RUNNER_ID}] run {run_id} crashed: {e}", file=sys.stderr)
        try:
            await flush()
            r = await http.post(
                f"/internal/runs/{run_id}/complete",
                json={
                    "runner_id": RUNNER_ID,
                    "status": "error",
                    "error": str(e)[:ERROR_MESSAGE_LIMIT],
                    "duration_s": round(duration_s, 3),
                },
            )
            r.raise_for_status()
        except Exception:
            pass  # complete 也失败：交给租约过期 + reaper 兜底
    finally:
        if not hb.done():
            hb.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await hb


async def main() -> None:
    headers = _auth_headers()
    if RUNNER_SECRET:
        print(f"[{RUNNER_ID}] auth: Bearer token configured")
    else:
        print(f"[{RUNNER_ID}] auth: none (dev mode)")
    print(f"[{RUNNER_ID}] polling {SERVER}")

    async with httpx.AsyncClient(
        base_url=SERVER,
        headers=headers,
        timeout=httpx.Timeout(10, read=120),
    ) as http:
        while True:
            try:
                r = await http.post("/internal/lease", json={"runner_id": RUNNER_ID})
                r.raise_for_status()
                lease = r.json().get("run")
            except Exception as e:
                print(f"[{RUNNER_ID}] lease failed ({e}); retrying", file=sys.stderr)
                await asyncio.sleep(POLL_IDLE_S)
                continue

            if lease is None:
                await asyncio.sleep(POLL_IDLE_S)
                continue
            await execute(http, lease)


if __name__ == "__main__":
    asyncio.run(main())
