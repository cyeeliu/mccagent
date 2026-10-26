"""后台任务测试：BackgroundTaskManager / RunBashTool / 工具 / AgentLoop 注入。"""

from __future__ import annotations

import asyncio

import pytest
from mcc_core import ToolContext
from mcc_harness import (
    BackgroundResult,
    BackgroundStatusTool,
    BackgroundTaskInfo,
    BackgroundTaskManager,
    CancelBackgroundTaskTool,
    format_background_results,
)
from mcc_tools import RunBashTool
from mcc_tools.sandbox import DirectExecutor


def _ctx(tmp_path):
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


# -- BackgroundTaskManager --


class TestBackgroundTaskManager:
    async def test_start_and_collect(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("echo hello", tmp_path, timeout=10)
        assert task_id.startswith("bg_")
        await asyncio.sleep(0.3)
        results = await mgr.collect()
        assert len(results) == 1
        assert results[0].task_id == task_id
        assert results[0].status == "completed"
        assert "hello" in results[0].output

    async def test_failed_command(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        await mgr.start("exit 1", tmp_path, timeout=10)
        await asyncio.sleep(0.3)
        results = await mgr.collect()
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].exit_code == 1

    async def test_collect_empty(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        assert await mgr.collect() == []

    async def test_multiple_concurrent(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor(), max_concurrent=4)
        ids = [await mgr.start(f"echo {i}", tmp_path, timeout=10) for i in range(4)]
        await asyncio.sleep(0.5)
        results = await mgr.collect()
        assert len(results) == 4
        collected_ids = {r.task_id for r in results}
        assert collected_ids == set(ids)

    async def test_task_ids_increment(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        id1 = await mgr.start("echo 1", tmp_path)
        id2 = await mgr.start("echo 2", tmp_path)
        assert id1 == "bg_0001"
        assert id2 == "bg_0002"
        await mgr.shutdown_all()

    async def test_list_tasks(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        await mgr.start("echo a", tmp_path)
        await mgr.start("echo b", tmp_path)
        tasks = mgr.list_tasks()
        assert len(tasks) == 2
        assert all(t.status in ("running", "completed") for t in tasks)
        await mgr.shutdown_all()

    async def test_get_status(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("echo x", tmp_path)
        info = mgr.get_status(task_id)
        assert info is not None
        assert info.task_id == task_id
        assert mgr.get_status("nonexistent") is None
        await mgr.shutdown_all()

    async def test_cancel(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("sleep 10", tmp_path, timeout=15)
        await asyncio.sleep(0.1)
        cancelled = await mgr.cancel(task_id)
        assert cancelled
        info = mgr.get_status(task_id)
        assert info.status == "cancelled"

    async def test_cancel_nonexistent(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        assert not await mgr.cancel("bg_9999")

    async def test_cancel_completed(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("echo done", tmp_path, timeout=5)
        await asyncio.sleep(0.3)
        assert not await mgr.cancel(task_id)

    async def test_shutdown_all(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        await mgr.start("sleep 5", tmp_path, timeout=10)
        await mgr.start("sleep 5", tmp_path, timeout=10)
        await mgr.shutdown_all()
        for info in mgr.list_tasks():
            assert info.status in ("cancelled", "completed", "failed")

    async def test_output_truncation(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        await mgr.start("yes | head -c 100000", tmp_path, timeout=10)
        await asyncio.sleep(0.5)
        results = await mgr.collect()
        assert len(results) == 1
        assert len(results[0].output) <= 50000


# -- format_background_results --


class TestFormatBackgroundResults:
    def test_empty(self):
        assert format_background_results([]) == ""

    def test_single(self):
        r = BackgroundResult(
            task_id="bg_0001", command="npm test",
            status="completed", output="all passed", exit_code=0,
        )
        result = format_background_results([r])
        assert "<background_result>" in result
        assert "bg_0001" in result
        assert "completed" in result
        assert "npm test" in result
        assert "all passed" in result

    def test_multiple(self):
        r1 = BackgroundResult("bg_1", "cmd1", "completed", "out1", 0)
        r2 = BackgroundResult("bg_2", "cmd2", "failed", "out2", 1)
        result = format_background_results([r1, r2])
        assert "bg_1" in result and "bg_2" in result
        assert "completed" in result and "failed" in result

    def test_summary_truncated(self):
        r = BackgroundResult("bg_1", "cmd", "completed", "x" * 1000, 0)
        result = format_background_results([r])
        assert "<summary>" in result
        assert "x" * 500 in result
        assert "x" * 1000 not in result


# -- RunBashTool with run_in_background --


class TestRunBashBackground:
    async def test_run_in_background_returns_task_id(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = RunBashTool(DirectExecutor(), mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"command": "echo hello", "run_in_background": True}, ctx
        )
        assert not result.is_error
        assert "bg_" in result.content
        assert "started" in result.content
        await mgr.shutdown_all()

    async def test_run_in_background_false_executes_sync(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = RunBashTool(DirectExecutor(), mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"command": "echo sync", "run_in_background": False}, ctx
        )
        assert not result.is_error
        assert "sync" in result.content

    async def test_no_background_manager_ignores_flag(self, tmp_path):
        tool = RunBashTool(DirectExecutor(), None)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"command": "echo fallback", "run_in_background": True}, ctx
        )
        assert not result.is_error
        assert "fallback" in result.content

    async def test_background_result_collected(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = RunBashTool(DirectExecutor(), mgr)
        ctx = _ctx(tmp_path)
        await tool.execute(
            {"command": "echo bg_result", "run_in_background": True}, ctx
        )
        await asyncio.sleep(0.3)
        results = await mgr.collect()
        assert len(results) == 1
        assert "bg_result" in results[0].output


# -- BackgroundStatusTool / CancelBackgroundTaskTool --


class TestBackgroundTools:
    async def test_status_all(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        await mgr.start("echo a", tmp_path)
        tool = BackgroundStatusTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "bg_" in result.content
        await mgr.shutdown_all()

    async def test_status_single(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("echo b", tmp_path)
        await asyncio.sleep(0.3)
        tool = BackgroundStatusTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"task_id": task_id}, ctx)
        assert not result.is_error
        assert task_id in result.content

    async def test_status_unknown(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = BackgroundStatusTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"task_id": "bg_9999"}, ctx)
        assert result.is_error

    async def test_status_empty(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = BackgroundStatusTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({}, ctx)
        assert "No background" in result.content

    async def test_cancel_tool(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        task_id = await mgr.start("sleep 10", tmp_path, timeout=15)
        await asyncio.sleep(0.1)
        tool = CancelBackgroundTaskTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"task_id": task_id}, ctx)
        assert not result.is_error
        assert "Cancelled" in result.content

    async def test_cancel_tool_missing_id(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = CancelBackgroundTaskTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({}, ctx)
        assert result.is_error

    async def test_cancel_tool_not_running(self, tmp_path):
        mgr = BackgroundTaskManager(DirectExecutor())
        tool = CancelBackgroundTaskTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"task_id": "bg_9999"}, ctx)
        assert result.is_error
