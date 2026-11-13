"""InteractiveRuntime：CLI 主 Agent 的交互式运行时。

多轮交互模型：start → idle → (run_turn → idle)* → stop → shutdown。
同一 Session 默认禁止并发执行多个 turn（asyncio.Lock 保护）。
通过 asyncio.Queue 桥接 AgentLoop.run_streaming 的 push 模型与
run_turn 的 pull 模型（async generator）。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from mcc_core import AgentEvent, Usage

from .base import (
    RuntimeConfig,
    RuntimeMode,
    RuntimeResult,
    RuntimeStatus,
    _terminal_result,
)

_ACTIVE_STATUSES = frozenset({
    RuntimeStatus.STARTING,
    RuntimeStatus.RUNNING,
    RuntimeStatus.IDLE,
    RuntimeStatus.PAUSED,
})


class InteractiveRuntime:
    """交互式运行时：包装 AgentLoop 的多轮对话。

    持有构建完成的 AgentLoop，通过 run_turn 驱动每一轮流式执行。
    同一实例禁止并发执行多个 turn（asyncio.Lock）。
    stop/cancel 在下一安全边界生效（cancel_event 协作取消）。
    """

    def __init__(
        self,
        agent: Any,
        *,
        runtime_id: str | None = None,
        session_id: str = "",
        agent_id: str = "",
    ) -> None:
        self._agent = agent
        self._runtime_id = runtime_id or f"interactive::{uuid4().hex[:12]}"
        self._session_id = session_id
        self._agent_id = agent_id or self._runtime_id
        self._status = RuntimeStatus.CREATED
        self._turn_lock = asyncio.Lock()
        self._closed = False
        self._cancel_event: asyncio.Event | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._started_at: float = 0.0
        self._result: RuntimeResult | None = None
        self._last_usage: Usage | None = None
        self._last_cost: float = 0.0

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status in _ACTIVE_STATUSES

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def config(self) -> RuntimeConfig:
        return RuntimeConfig(
            name=self._agent_id,
            mode=RuntimeMode.INTERACTIVE,
            session_id=self._session_id,
            agent_id=self._agent_id,
        )

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        """标记运行时就绪，进入 IDLE 状态等待 turn。

        task 参数保留用于满足 AgentRuntime 协议签名，
        实际执行通过 run_turn 驱动。
        """
        if self._closed:
            raise RuntimeError(
                f"interactive runtime {self._runtime_id} is closed"
            )
        if self._status is RuntimeStatus.RUNNING:
            raise RuntimeError(
                f"interactive runtime {self._runtime_id} already running"
            )
        if self._status is RuntimeStatus.IDLE:
            raise RuntimeError(
                f"interactive runtime {self._runtime_id} already started"
            )
        self._started_at = time.monotonic()
        self._result = None
        self._status = RuntimeStatus.IDLE

    async def run_turn(self, prompt: str) -> AsyncIterator[AgentEvent]:
        """执行一个交互轮次，产出 AgentEvent 事件流。

        同一实例禁止并发执行多个 turn（asyncio.Lock 保护）。
        cancel_event 在下一迭代边界协作取消。
        """
        if self._closed:
            raise RuntimeError(
                f"interactive runtime {self._runtime_id} is closed"
            )
        async with self._turn_lock:
            self._status = RuntimeStatus.RUNNING
            self._cancel_event = asyncio.Event()
            queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

            async def worker() -> None:
                try:
                    run = await self._agent.run_streaming(
                        prompt,
                        queue,
                        cancel_event=self._cancel_event,
                    )
                    self._last_usage = run.total_usage
                    self._last_cost = run.cost_usd
                except Exception as exc:
                    from mcc_core import Error as ErrorEvent

                    await queue.put(ErrorEvent(text=str(exc)))
                finally:
                    await queue.put(None)

            task = asyncio.create_task(worker())
            self._current_task = task
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
                self._current_task = None
                self._cancel_event = None
                if not self._closed:
                    self._status = RuntimeStatus.IDLE

    async def stop(self, *, timeout: float = 5.0) -> None:
        """关闭运行时，取消当前 turn 并等待退出。"""
        if self._closed:
            return
        self._closed = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._current_task, timeout=timeout)
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        """发送取消信号（协作式，下一迭代边界生效）。"""
        if self._cancel_event is not None:
            self._cancel_event.set()

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        """等待运行时到达终态。

        InteractiveRuntime 需要外部调用 stop() 后才会到达终态。
        timeout 到期时抛出 WaitTimeoutError。
        """
        from .base import WaitTimeoutError

        if self._result is not None:
            return self._result

        if not self._closed:
            if timeout is not None:
                raise WaitTimeoutError(
                    f"wait_done timed out after {timeout}s for {self._runtime_id}"
                )
            return _terminal_result(
                self.runtime_id,
                self._status,
                summary="not closed",
                agent_id=self._agent_id,
            )

        self._result = _terminal_result(
            self.runtime_id,
            RuntimeStatus.SHUTDOWN,
            summary="shutdown",
            started_at=self._started_at,
            agent_id=self._agent_id,
            usage=self._last_usage,
            cost_usd=self._last_cost,
        )
        return self._result
