"""TaskDispatcher：统一任务派发器。

Phase 6 交付物（G-05）：将任务派发逻辑从 Orchestrator 内聚中抽出。
TaskDispatcher 负责：
- 从 TaskStore 取任务，创建 Attempt
- 通过 RuntimeCoordinator 找到 Runtime 并 submit
- 监听 Runtime 事件，等待任务完成
- 结果先写 TaskStore，再发完成事件
- 支持幂等 submit（同一 idempotency_key 不重复执行）
- 支持 retry（创建新 attempt_id）

不负责规划任务或选择 Agent（由 Planner/Scheduler 负责）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from uuid import uuid4

from mcc_core.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeTask,
    TaskContext,
    TaskHandle,
    TaskStatus,
    TaskStoreProtocol,
    TokenUsage,
)
from mcc_core.runtime.task_record import (
    AttemptStatus,
    TaskAttempt,
)

from .coordinator import RuntimeCoordinator

logger = logging.getLogger(__name__)

_TASK_DONE_EVENTS = frozenset(
    {
        RuntimeEventType.TASK_COMPLETED,
        RuntimeEventType.TASK_FAILED,
        RuntimeEventType.TASK_CANCELLED,
        RuntimeEventType.TASK_TIMED_OUT,
    }
)

_ATTEMPT_STATUS_TO_EVENT: dict[AttemptStatus, RuntimeEventType] = {
    AttemptStatus.SUCCEEDED: RuntimeEventType.TASK_COMPLETED,
    AttemptStatus.FAILED: RuntimeEventType.TASK_FAILED,
    AttemptStatus.TIMED_OUT: RuntimeEventType.TASK_TIMED_OUT,
    AttemptStatus.CANCELLED: RuntimeEventType.TASK_CANCELLED,
}


class DispatchError(RuntimeError):
    """任务派发失败。"""


class TaskDispatcher:
    """统一任务派发器。

    从 TaskStore 取任务，通过 RuntimeCoordinator 分配给 Runtime 执行，
    结果回写 TaskStore。支持幂等 submit 和 retry。

    用法：
        dispatcher = TaskDispatcher(task_store, coordinator)
        handle = await dispatcher.dispatch(task_id, agent_id, runtime_id="rt-1")
        result = await dispatcher.wait_for_result(handle, timeout=30.0)
    """

    def __init__(
        self,
        task_store: TaskStoreProtocol,
        coordinator: RuntimeCoordinator,
    ) -> None:
        self._store = task_store
        self._coordinator = coordinator
        self._pending: dict[str, asyncio.Future[RuntimeEvent]] = {}

    async def dispatch(
        self,
        task_id: str,
        agent_id: str,
        *,
        runtime_id: str,
        objective: str | None = None,
        input_data: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskHandle:
        """派发任务到指定 Runtime。

        幂等：同一 idempotency_key 不会创建新 Attempt。
        事务：submit 失败时 Attempt 标记为 FAILED。
        """
        task = await self._store.get_task(task_id)
        if task is None:
            from mcc_core.runtime.errors import TaskNotFoundError

            raise TaskNotFoundError(task_id)

        # 幂等检查：依赖 TaskStore 的 idempotency_key 机制（持久化、跨进程），
        # 而非内存 set。若已有同 key 的 attempt，直接返回 handle 不重复 submit。
        if idempotency_key is not None:
            existing = await self._store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                logger.info(
                    "idempotent dispatch: task=%s attempt=%s already exists",
                    task_id,
                    existing.attempt_id,
                )
                return TaskHandle(
                    task_id=task_id,
                    attempt_id=existing.attempt_id,
                    runtime_id=existing.runtime_id or runtime_id,
                )

        attempt = await self._store.create_attempt(
            task_id=task_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
            idempotency_key=idempotency_key,
        )

        # create_attempt 内部可能因竞态返回已有非 PENDING attempt，同样视为幂等
        if attempt.status is not AttemptStatus.PENDING:
            logger.info(
                "idempotent dispatch: task=%s attempt=%s already exists",
                task_id,
                attempt.attempt_id,
            )
            return TaskHandle(
                task_id=task_id,
                attempt_id=attempt.attempt_id,
                runtime_id=attempt.runtime_id or runtime_id,
            )

        runtime = self._coordinator.get(runtime_id)
        if runtime is None:
            await self._store.fail_attempt(
                attempt.attempt_id,
                error=f"runtime not found: {runtime_id}",
            )
            raise DispatchError(f"runtime not found: {runtime_id}")

        rt_objective = objective or task.subject
        rt_input = input_data or {}
        rt_task = RuntimeTask(
            task_id=task_id,
            attempt_id=attempt.attempt_id,
            agent_id=agent_id,
            objective=rt_objective,
            input=rt_input,
            run_id=run_id or task.run_id,
            timeout_seconds=timeout_seconds,
            idempotency_key=attempt.idempotency_key,
            context=TaskContext(session_id=run_id),
        )

        try:
            if task.status != TaskStatus.DISPATCHED:
                await self._store.transition_task(
                    task_id,
                    task.status,
                    TaskStatus.DISPATCHED,
                    agent_id=agent_id,
                )
            # PENDING → RUNNING：记录 attempt 已进入执行阶段
            await self._store.start_attempt(attempt.attempt_id)
            handle = await runtime.submit(rt_task)
            await self._store.transition_task(
                task_id,
                TaskStatus.DISPATCHED,
                TaskStatus.ACCEPTED,
                agent_id=agent_id,
            )
            return handle
        except asyncio.CancelledError:
            await self._store.fail_attempt(
                attempt.attempt_id,
                error="cancelled",
                status=AttemptStatus.CANCELLED,
            )
            raise
        except Exception as exc:
            await self._store.fail_attempt(
                attempt.attempt_id,
                error=str(exc),
            )
            logger.exception(
                "dispatch failed: task=%s attempt=%s",
                task_id,
                attempt.attempt_id,
            )
            raise DispatchError(
                f"dispatch failed for task {task_id}: {exc}",
            ) from exc

    async def wait_for_result(
        self,
        handle: TaskHandle,
        *,
        timeout: float | None = None,
    ) -> RuntimeEvent:
        """等待任务完成，返回完成事件。

        P0-01: 先订阅事件流（注册 listener），再查 TaskStore，消除竞态窗口。
        若 attempt 已终态，直接返回合成事件（不依赖事件流）。
        否则监听 Runtime 事件流，直到收到 TASK_COMPLETED / TASK_FAILED /
        TASK_CANCELLED / TASK_TIMED_OUT 事件。
        超时仅抛 TaskWaitTimeoutError，不修改 attempt 状态。
        """
        runtime = self._coordinator.get(handle.runtime_id)
        if runtime is None:
            raise DispatchError(
                f"runtime not found: {handle.runtime_id}",
            )

        # P0-01: 先订阅事件流，再查 TaskStore，消除竞态窗口
        event_iter = runtime.subscribe_events(
            task_id=handle.task_id,
            attempt_id=handle.attempt_id,
        )

        priming: asyncio.Task | None = None
        try:
            # Prime generator 以确保 listener 注册到 runtime 的事件队列
            priming = asyncio.ensure_future(event_iter.__anext__())
            await asyncio.sleep(0)  # 让 priming 执行到 queue.get() 挂起

            # listener 已注册，安全查 TaskStore
            attempt = await self._store.get_attempt(handle.attempt_id)
            if attempt is not None and attempt.is_terminal:
                return self._synthetic_event(handle, attempt)

            # 未终态，等待事件
            async def _wait() -> RuntimeEvent:
                # priming 可能已拿到事件
                first_event = await priming
                if first_event.event_type in _TASK_DONE_EVENTS:
                    await self._record_result(handle, first_event)
                    return first_event
                async for event in event_iter:
                    if event.event_type in _TASK_DONE_EVENTS:
                        await self._record_result(handle, event)
                        return event
                raise DispatchError(
                    f"event stream ended without completion: "
                    f"task={handle.task_id} attempt={handle.attempt_id}",
                )

            if timeout is not None:
                return await asyncio.wait_for(_wait(), timeout=timeout)
            return await _wait()
        except TimeoutError:
            from mcc_core.runtime.errors import TaskWaitTimeoutError

            # 超时后取消真实执行，确保旧 attempt 终止后才允许 retry
            with contextlib.suppress(Exception):
                cancel = getattr(runtime, "cancel_task", None)
                if cancel is not None:
                    await cancel(
                        handle.task_id,
                        handle.attempt_id,
                        reason="wait timeout",
                    )
            raise TaskWaitTimeoutError(handle.task_id, timeout or 0.0) from None
        finally:
            if priming is not None and not priming.done():
                priming.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await priming
            with contextlib.suppress(Exception):
                await event_iter.aclose()

    def _synthetic_event(
        self,
        handle: TaskHandle,
        attempt: TaskAttempt,
    ) -> RuntimeEvent:
        """从 TaskAttempt 终态构造合成 RuntimeEvent。"""
        event_type = _ATTEMPT_STATUS_TO_EVENT.get(attempt.status)
        if event_type is None:
            raise DispatchError(
                f"attempt {attempt.attempt_id} has non-terminal status {attempt.status}",
            )
        payload: dict[str, object] = {}
        if attempt.summary:
            payload["summary"] = attempt.summary
        if attempt.error:
            payload["error"] = attempt.error
        return RuntimeEvent(
            event_id=f"evt-{uuid4().hex[:12]}",
            event_type=event_type,
            runtime_id=handle.runtime_id,
            task_id=handle.task_id,
            attempt_id=handle.attempt_id,
            sequence=0,
            occurred_at=datetime.now(UTC),
            payload=payload,
        )

    async def _record_result(
        self,
        handle: TaskHandle,
        event: RuntimeEvent,
    ) -> None:
        """将结果写入 TaskStore。

        若 attempt 已终态（Runtime 已写入完整结果），则跳过不覆盖。
        从事件 payload 读取完整字段（output / usage / cost_usd）（P0-03）。
        """
        attempt = await self._store.get_attempt(handle.attempt_id)
        if attempt is not None and attempt.is_terminal:
            return

        if event.event_type is RuntimeEventType.TASK_COMPLETED:
            usage = self._payload_usage(event.payload)
            cost_raw = event.payload.get("cost_usd")
            cost = float(cost_raw) if cost_raw is not None else None
            await self._store.complete_attempt(
                handle.attempt_id,
                summary=str(event.payload.get("summary", "")),
                output=event.payload.get("output"),
                usage=usage,
                cost_usd=cost,
            )
        elif event.event_type is RuntimeEventType.TASK_FAILED:
            await self._store.fail_attempt(
                handle.attempt_id,
                error=str(event.payload.get("error", "unknown")),
            )
        elif event.event_type is RuntimeEventType.TASK_CANCELLED:
            await self._store.fail_attempt(
                handle.attempt_id,
                error="cancelled",
                status=AttemptStatus.CANCELLED,
            )
        elif event.event_type is RuntimeEventType.TASK_TIMED_OUT:
            await self._store.fail_attempt(
                handle.attempt_id,
                error="timed out",
                status=AttemptStatus.TIMED_OUT,
            )

    @staticmethod
    def _payload_usage(
        payload: dict[str, object],
    ) -> TokenUsage | None:
        """从 payload 中的 usage dict 重建 TokenUsage（P0-03）。"""
        raw = payload.get("usage")
        if raw is None or not isinstance(raw, dict):
            return None
        return TokenUsage(
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens", 0)),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens", 0)),
        )

    async def retry(
        self,
        task_id: str,
        *,
        runtime_id: str,
        agent_id: str | None = None,
    ) -> TaskHandle:
        """重试任务：创建新 Attempt 并重新派发。

        task_id 保持不变，生成新的 attempt_id。
        需要任务未超过 max_attempts。
        """
        task = await self._store.get_task(task_id)
        if task is None:
            from mcc_core.runtime.errors import TaskNotFoundError

            raise TaskNotFoundError(task_id)
        if not task.can_retry:
            raise DispatchError(
                f"task {task_id} exceeded max_attempts ({task.max_attempts})",
            )

        resolve_agent = agent_id or task.agent_id or "retry-agent"
        return await self.dispatch(
            task_id,
            resolve_agent,
            runtime_id=runtime_id,
            idempotency_key=None,
        )

    async def fail_attempt(
        self,
        handle: TaskHandle,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> None:
        """标记 attempt 为失败 / 超时 / 取消。

        供 Orchestrator 在超时隔离或级联取消时调用，
        确保旧 attempt 不会永远停留在 RUNNING 状态。
        """
        await self._store.fail_attempt(
            handle.attempt_id,
            error=error,
            status=status,
        )

    async def dispatch_batch(
        self,
        tasks: list[tuple[str, str, str]],
        *,
        timeout_per_task: float | None = None,
    ) -> list[TaskHandle | Exception]:
        """批量派发任务，返回每个任务的结果（TaskHandle 或 Exception）。

        tasks: [(task_id, agent_id, runtime_id), ...]
        单个任务失败不阻止其他任务。
        """

        async def _safe_dispatch(
            task_id: str,
            agent_id: str,
            rt_id: str,
        ) -> TaskHandle | Exception:
            try:
                return await self.dispatch(task_id, agent_id, runtime_id=rt_id)
            except Exception as exc:
                return exc

        return await asyncio.gather(
            *(_safe_dispatch(tid, aid, rid) for tid, aid, rid in tasks),
        )


__all__ = [
    "TaskDispatcher",
    "DispatchError",
]
