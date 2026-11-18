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
import logging

from mcc_core.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeTask,
    TaskContext,
    TaskHandle,
    TaskStatus,
    TaskStoreProtocol,
)
from mcc_core.runtime.task_record import (
    AttemptStatus,
)

from .coordinator import RuntimeCoordinator

logger = logging.getLogger(__name__)

_TASK_DONE_EVENTS = frozenset({
    RuntimeEventType.TASK_COMPLETED,
    RuntimeEventType.TASK_FAILED,
    RuntimeEventType.TASK_CANCELLED,
    RuntimeEventType.TASK_TIMED_OUT,
})


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
        self._dispatched: set[str] = set()

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

        attempt = await self._store.create_attempt(
            task_id=task_id,
            agent_id=agent_id,
            runtime_id=runtime_id,
            idempotency_key=idempotency_key,
        )

        if attempt.status is not AttemptStatus.PENDING or attempt.attempt_id in self._dispatched:
            logger.info(
                "idempotent dispatch: task=%s attempt=%s already exists",
                task_id,
                attempt.attempt_id,
            )
            return TaskHandle(
                task_id=task_id,
                attempt_id=attempt.attempt_id,
                runtime_id=runtime_id,
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
            handle = await runtime.submit(rt_task)
            self._dispatched.add(attempt.attempt_id)
            await self._store.transition_task(
                task_id,
                TaskStatus.DISPATCHED,
                TaskStatus.ACCEPTED,
                agent_id=agent_id,
            )
            return handle
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

        监听 Runtime 事件流，直到收到 TASK_COMPLETED / TASK_FAILED /
        TASK_CANCELLED / TASK_TIMED_OUT 事件。
        结果先写 TaskStore，再返回事件。
        """
        runtime = self._coordinator.get(handle.runtime_id)
        if runtime is None:
            raise DispatchError(
                f"runtime not found: {handle.runtime_id}",
            )

        event_iter = runtime.subscribe_events(
            task_id=handle.task_id,
            attempt_id=handle.attempt_id,
        )

        try:
            async def _wait() -> RuntimeEvent:
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
            from mcc_core.runtime.errors import TaskTimeoutError

            await self._store.fail_attempt(
                handle.attempt_id,
                error=f"wait_for_result timeout after {timeout}s",
            )
            raise TaskTimeoutError(handle.task_id, handle.attempt_id) from None

    async def _record_result(
        self,
        handle: TaskHandle,
        event: RuntimeEvent,
    ) -> None:
        """将结果写入 TaskStore。"""
        if event.event_type is RuntimeEventType.TASK_COMPLETED:
            await self._store.complete_attempt(
                handle.attempt_id,
                summary=str(event.payload.get("summary", "")),
                output=event.payload.get("output"),
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
            )
        elif event.event_type is RuntimeEventType.TASK_TIMED_OUT:
            await self._store.fail_attempt(
                handle.attempt_id,
                error="timed out",
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
            *(
                _safe_dispatch(tid, aid, rid)
                for tid, aid, rid in tasks
            ),
        )


__all__ = [
    "TaskDispatcher",
    "DispatchError",
]
