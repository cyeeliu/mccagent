"""Runner 主循环：lease → 执行 adapter → 批量上报事件 → complete。

要点：
- Bearer token 鉴权（RUNNER_SECRET），未配置时为开发模式
- 心跳续租后台任务；判死时立即取消执行流止损（不等事件间隙）
- 事件缓冲按 FLUSH_INTERVAL_S 批量上报（JSONL），失败回灌重试
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
HEARTBEAT_RETRY_S = 5.0  # 心跳失败后的短间隔重试（远小于租约 TTL）
HEARTBEAT_MAX_FAILURES = 3  # 连续失败判死阈值
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


def _has_signal(snap: TraceSnapshot) -> bool:
    """轨迹是否包含可评分的实质内容（工具调用或非空模型输出）。"""
    return bool(snap.tool_calls) or bool("".join(snap.text_outputs).strip())


def _final_status_score(
    final: RunFinished | None,
    snap: TraceSnapshot,
    expectations: list[dict],
) -> tuple[str, float | None]:
    """计算 complete 上报的 status 与 score。

    adapter 返回的 RunFinished.status 是执行状态（error 只代表执行中
    出过未恢复的异常，不代表轨迹不可评）。有 expectations 且轨迹有
    实质内容（工具调用或模型输出）时按评分覆盖——no_tool_errors
    断言会正确惩罚带错的轨迹；零产出的 error（如凭证未配、LLM
    首调即挂）保留 error 状态不评分，避免把基础设施故障误判成
    Agent 低分。
    """
    if final and expectations and _has_signal(snap):
        result = compute_score(snap, expectations)
        return result.status, result.score
    if final:
        return final.status, final.score
    return "error", None


def _cancel_victim(victim: asyncio.Task | None) -> None:
    """心跳判死后主动取消执行流。"""
    if victim is not None and not victim.done():
        victim.cancel()


async def heartbeat_loop(
    http: httpx.AsyncClient, run_id: str, victim: asyncio.Task | None = None
) -> None:
    """后台心跳：租约被 reaper 回收时主动失败。

    单次网络失败不判死（心跳对传输抖动容错，短间隔重试，连续
    HEARTBEAT_MAX_FAILURES 次失败才放弃）——server 明确返回
    alive=false（租约被回收）才立即失败。

    判死时若持有执行任务引用（victim）则立即将其取消：执行流的
    租约检查点是事件驱动的，工具执行或慢 LLM 期间可能长时间没有
    事件，若不主动取消，主流程会在租约已丢失后继续空跑（继续烧
    LLM 成本），且收尾的 complete 必然被 server 拒绝——reaper 已
    把 run 重排（retries 未耗尽时置回 queued），后续由其他循环
    重新领走执行，造成「轨迹已结束、run 却还在跑」的矛盾。
    """
    failures = 0
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S if failures == 0 else HEARTBEAT_RETRY_S)
        try:
            r = await http.post(
                f"/internal/runs/{run_id}/heartbeat",
                json={"runner_id": RUNNER_ID},
            )
            r.raise_for_status()
        except Exception as e:
            failures += 1
            if failures >= HEARTBEAT_MAX_FAILURES:
                _cancel_victim(victim)
                raise RuntimeError(f"heartbeat failed {failures} times: {e}") from e
            print(
                f"[{RUNNER_ID}] heartbeat for {run_id} failed "
                f"({failures}/{HEARTBEAT_MAX_FAILURES}): {e}",
                file=sys.stderr,
            )
            continue
        failures = 0
        if not r.json().get("alive"):
            # 租约被回收（多半是本进程卡顿太久）
            _cancel_victim(victim)
            raise RuntimeError("lease lost")


async def execute(http: httpx.AsyncClient, lease: dict) -> None:
    """执行单个 run：跑 adapter、攒事件批量上报、评分、complete。

    执行主体跑在子任务里，心跳任务持有其引用：租约判死时立即
    取消执行流止损（不等下一个事件——工具执行或慢 LLM 期间可能
    长时间没有事件，继续空跑只会浪费 LLM 成本且收尾必然失败）。
    """
    run_id = lease["run_id"]
    expectations: list[dict] = lease.get("expectations") or []

    buf: list[str] = []
    snap = TraceSnapshot()
    final: RunFinished | None = None
    t_start = time.monotonic()

    async def flush() -> None:
        if not buf:
            return
        lines, buf[:] = "\n".join(buf), []
        try:
            r = await http.post(f"/internal/runs/{run_id}/events", content=lines)
            r.raise_for_status()
        except Exception:
            # 上报失败回灌缓冲：收尾路径重试，事件不因瞬时故障丢失
            buf[:0] = lines.splitlines()
            raise

    def ensure_heartbeat_alive() -> None:
        if hb.done():
            hb.result()

    async def body() -> None:
        nonlocal final
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

        status, score_val = _final_status_score(final, snap, expectations)

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

    async def report_crash(exc_text: str) -> None:
        """收尾路径：上报已产出的结果，失败时给出可定位的诊断日志。"""
        duration_s = time.monotonic() - t_start
        try:
            await flush()
            if final is not None:
                status, score_val = _final_status_score(final, snap, expectations)
                error_text = f"completed with finish error: {exc_text}"
            else:
                status, score_val = "error", None
                error_text = exc_text[:ERROR_MESSAGE_LIMIT]
            r = await http.post(
                f"/internal/runs/{run_id}/complete",
                json={
                    "runner_id": RUNNER_ID,
                    "status": status,
                    "score": score_val,
                    "cost_usd": final.cost_usd if final is not None else None,
                    "turns": final.turns if final is not None else None,
                    "duration_s": round(duration_s, 3),
                    "error": error_text[:ERROR_MESSAGE_LIMIT],
                },
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                # run 已被 reaper 回收重排（会重新执行），complete 被拒
                print(
                    f"[{RUNNER_ID}] run {run_id} complete rejected (409): "
                    f"lease already reclaimed by reaper, run will be retried",
                    file=sys.stderr,
                )
            else:
                print(f"[{RUNNER_ID}] run {run_id} complete failed: {e}", file=sys.stderr)
        except Exception as e:
            # complete 也失败：交给租约过期 + reaper 兜底
            print(f"[{RUNNER_ID}] run {run_id} complete failed: {e}", file=sys.stderr)

    body_task = asyncio.create_task(body())
    hb = asyncio.create_task(heartbeat_loop(http, run_id, body_task))
    try:
        await body_task
    except asyncio.CancelledError:
        cur = asyncio.current_task()
        if body_task.cancelled() and cur is not None and cur.cancelling() == 0:
            # 心跳判死主动取消了执行流：止损并走收尾上报
            hb_exc = hb.exception() if hb.done() and not hb.cancelled() else None
            exc_text = str(hb_exc) if hb_exc else "lease lost"
            print(
                f"[{RUNNER_ID}] run {run_id} aborted: {exc_text}; "
                f"cancelling execution to stop burning LLM cost",
                file=sys.stderr,
            )
            await report_crash(exc_text)
        else:
            # execute 自身被取消（如进程停机），向上传播
            raise
    except Exception as e:
        # 单 run 失败隔离：上报后继续下一个任务。
        # 评分保全：run_finished 已产出（adapter 走完整个生命周期）时，
        # 收尾阶段的异常（心跳断连、上报抖动等）不应吞掉已完成的
        # 评分结果——轨迹与 run 状态会因此互相矛盾；此时 status/score
        # 仍按评分逻辑取值，异常摘要附在 error 字段。
        print(f"[{RUNNER_ID}] run {run_id} crashed: {e}", file=sys.stderr)
        await report_crash(str(e))
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
