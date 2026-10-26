"""后台任务管理：异步执行长耗时 shell 命令，不阻塞主循环。

命令在 asyncio.Task 中通过 SandboxExecutor 执行（沙箱隔离），
完成后结果存入 ready 队列，AgentLoop 每轮迭代首 collect 并注入为
user 消息，使模型能感知后台任务完成。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput
from mcc_tools.sandbox import DirectExecutor, ExecTimeout, SandboxExecutor


@dataclass
class BackgroundTaskInfo:
    """后台任务元信息快照。"""

    task_id: str
    command: str
    status: str
    started_at: float
    finished_at: float | None = None


@dataclass
class BackgroundResult:
    """后台任务完成结果。"""

    task_id: str
    command: str
    status: str
    output: str
    exit_code: int | None


class BackgroundTaskManager:
    """后台任务管理器：asyncio.Task 驱动的异步命令执行。

    ``start`` 创建 asyncio.Task 执行命令（沙箱隔离），
    完成后结果存入 ready 队列；``collect`` 非阻塞批量取出；
    AgentLoop 每轮迭代首调用 collect 并注入为 user 消息。
    """

    def __init__(
        self,
        executor: SandboxExecutor | None = None,
        max_concurrent: int = 8,
        default_timeout: float = 120.0,
    ) -> None:
        self._executor = executor or DirectExecutor()
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._infos: dict[str, BackgroundTaskInfo] = {}
        self._results: dict[str, BackgroundResult] = {}
        self._ready: asyncio.Queue[str] = asyncio.Queue()
        self._counter = 0
        self._lock = asyncio.Lock()

    async def start(
        self,
        command: str,
        cwd: Path,
        timeout: float | None = None,
    ) -> str:
        """启动后台命令，返回 task_id。"""
        async with self._lock:
            self._counter += 1
            task_id = f"bg_{self._counter:04d}"
            self._infos[task_id] = BackgroundTaskInfo(
                task_id=task_id,
                command=command,
                status="running",
                started_at=time.time(),
            )

        task = asyncio.create_task(
            self._run(task_id, command, cwd, timeout or self._default_timeout)
        )
        self._tasks[task_id] = task
        return task_id

    async def _run(
        self,
        task_id: str,
        command: str,
        cwd: Path,
        timeout: float,
    ) -> None:
        async with self._semaphore:
            try:
                result = await self._executor.run(command, cwd, timeout)
                output = (
                    result.stdout.decode("utf-8", errors="replace")
                    + result.stderr.decode("utf-8", errors="replace")
                ).strip()
                exit_code = result.returncode
                status = "completed" if exit_code == 0 else "failed"
            except ExecTimeout:
                output = f"Error: command timed out after {timeout}s"
                exit_code = None
                status = "failed"
            except Exception as e:
                output = f"Error: {type(e).__name__}: {e}"
                exit_code = None
                status = "failed"

        async with self._lock:
            info = self._infos.get(task_id)
            if info is None:
                return
            info.status = status
            info.finished_at = time.time()
            self._results[task_id] = BackgroundResult(
                task_id=task_id,
                command=command,
                status=status,
                output=output[:50000],
                exit_code=exit_code,
            )
            await self._ready.put(task_id)

    async def collect(self) -> list[BackgroundResult]:
        """非阻塞批量取出已完成任务结果。"""
        results: list[BackgroundResult] = []
        while not self._ready.empty():
            task_id = self._ready.get_nowait()
            result = self._results.pop(task_id, None)
            if result is not None:
                results.append(result)
        return results

    def list_tasks(self) -> list[BackgroundTaskInfo]:
        """列出所有任务（含运行中和已完成的元信息）。"""
        return list(self._infos.values())

    def get_status(self, task_id: str) -> BackgroundTaskInfo | None:
        return self._infos.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        """取消运行中的后台任务。"""
        task = self._tasks.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        async with self._lock:
            info = self._infos.get(task_id)
            if info and info.status == "running":
                info.status = "cancelled"
                info.finished_at = time.time()
        return True

    async def shutdown_all(self) -> None:
        """取消所有运行中的后台任务并等待退出。"""
        running = [tid for tid, t in self._tasks.items() if not t.done()]
        for tid in running:
            await self.cancel(tid)
        pending = [self._tasks[tid] for tid in running if not self._tasks[tid].done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def format_background_results(results: list[BackgroundResult]) -> str:
    """将后台任务结果格式化为注入 prompt 的通知文本。"""
    parts = []
    for r in results:
        parts.append(
            f"<background_result>\n"
            f"  <task_id>{r.task_id}</task_id>\n"
            f"  <status>{r.status}</status>\n"
            f"  <command>{r.command}</command>\n"
            f"  <summary>{r.output[:500]}</summary>\n"
            f"</background_result>"
        )
    return "\n".join(parts)


# -- 工具 --


class BackgroundStatusTool:
    """background_status：查询后台任务状态。"""

    def __init__(self, manager: BackgroundTaskManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "background_status"

    @property
    def description(self) -> str:
        return "Check status of background tasks. Pass task_id for one, omit for all."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id")
        if task_id:
            info = self._manager.get_status(str(task_id))
            if info is None:
                return ToolOutput.err(f"unknown task: {task_id}")
            return ToolOutput.ok(_format_info(info))
        infos = self._manager.list_tasks()
        if not infos:
            return ToolOutput.ok("No background tasks")
        return ToolOutput.ok("\n".join(_format_info(i) for i in infos))


class CancelBackgroundTaskTool:
    """cancel_background：取消运行中的后台任务。"""

    def __init__(self, manager: BackgroundTaskManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "cancel_background"

    @property
    def description(self) -> str:
        return "Cancel a running background task by task_id."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = str(input.get("task_id", ""))
        if not task_id:
            return ToolOutput.err("'task_id' is required")
        cancelled = await self._manager.cancel(task_id)
        if not cancelled:
            return ToolOutput.err(f"task not running or not found: {task_id}")
        return ToolOutput.ok(f"Cancelled {task_id}")


def _format_info(info: BackgroundTaskInfo) -> str:
    line = f"- {info.task_id} [{info.status}] {info.command[:60]}"
    if info.finished_at:
        elapsed = info.finished_at - info.started_at
        line += f" ({elapsed:.1f}s)"
    return line


__all__ = [
    "BackgroundResult",
    "BackgroundStatusTool",
    "BackgroundTaskInfo",
    "BackgroundTaskManager",
    "CancelBackgroundTaskTool",
    "format_background_results",
]
