"""PR-12: TeammateRuntime 纯消息适配化（P1-09）

验证 TeammateRuntime 不再包含旧执行路径：
- _do_work() / _dequeue_assignment() / _set_task_future() 等方法已删除
- _task_futures / _assignment_queue 属性已删除
- _run_loop() 不调用 _do_work() 或 run_streaming()
- send_message() 不触发旧 AgentLoop 执行
- "message" kind 消息路由到 _instruction_queue 而非触发执行
- build_agent_executor() 仍然可用（供 PersistentRuntimeV2 装配）
- TeammateRuntime 仍可正常 start/stop
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from mcc_harness import LEAD_NAME, MessageBus, PlanStore, TeammateRuntime, TeammateSpec
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry

# -- 辅助函数 --


def _make_teammate(tmp_path, llm=None, name="rt_test"):
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    registry = ToolRegistry()
    spec = TeammateSpec(
        name=name,
        role="coder",
        system="test",
        model="claude-haiku-4-5",
    )
    rt = TeammateRuntime(
        spec=spec,
        llm=llm or MockLlm([single_text_turn("done")]),
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        base_dir=tmp_path,
    )
    return rt, bus


# -- 架构守卫：旧执行路径方法已删除 --


class TestOldExecutionPathRemoved:
    """TeammateRuntime 不再包含旧执行路径的方法。"""

    def test_do_work_removed(self):
        assert not hasattr(TeammateRuntime, "_do_work"), (
            "TeammateRuntime._do_work 应已删除"
        )

    def test_dequeue_assignment_removed(self):
        assert not hasattr(TeammateRuntime, "_dequeue_assignment"), (
            "TeammateRuntime._dequeue_assignment 应已删除"
        )

    def test_set_task_future_removed(self):
        assert not hasattr(TeammateRuntime, "_set_task_future"), (
            "TeammateRuntime._set_task_future 应已删除"
        )

    def test_get_task_future_removed(self):
        assert not hasattr(TeammateRuntime, "get_task_future"), (
            "TeammateRuntime.get_task_future 应已删除"
        )

    def test_set_task_future_method_removed(self):
        assert not hasattr(TeammateRuntime, "set_task_future"), (
            "TeammateRuntime.set_task_future 应已删除"
        )

    def test_try_claim_task_removed(self):
        assert not hasattr(TeammateRuntime, "_try_claim_task"), (
            "TeammateRuntime._try_claim_task 应已删除"
        )

    def test_submit_and_wait_plan_removed(self):
        assert not hasattr(TeammateRuntime, "_submit_and_wait_plan"), (
            "TeammateRuntime._submit_and_wait_plan 应已删除"
        )

    def test_prepare_worktree_removed(self):
        assert not hasattr(TeammateRuntime, "_prepare_worktree"), (
            "TeammateRuntime._prepare_worktree 应已删除"
        )

    def test_cleanup_worktree_removed(self):
        assert not hasattr(TeammateRuntime, "_cleanup_worktree"), (
            "TeammateRuntime._cleanup_worktree 应已删除"
        )


# -- 架构守卫：旧属性已删除 --


class TestOldAttributesRemoved:
    """TeammateRuntime 不再包含旧执行路径的实例属性。"""

    def test_no_task_futures_attr(self, tmp_path):
        rt, _ = _make_teammate(tmp_path)
        assert not hasattr(rt, "_task_futures"), (
            "TeammateRuntime 不应包含 _task_futures 属性"
        )

    def test_no_assignment_queue_attr(self, tmp_path):
        rt, _ = _make_teammate(tmp_path)
        assert not hasattr(rt, "_assignment_queue"), (
            "TeammateRuntime 不应包含 _assignment_queue 属性"
        )


# -- 架构守卫：_run_loop 源码不含旧执行调用 --


class TestRunLoopSourceGuard:
    """_run_loop() 源码不含 _do_work / run_streaming 调用。"""

    def test_run_loop_no_do_work_call(self):
        source = inspect.getsource(TeammateRuntime._run_loop)
        assert "_do_work" not in source, (
            "_run_loop 源码不应包含 _do_work 调用"
        )

    def test_run_loop_no_run_streaming_call(self):
        source = inspect.getsource(TeammateRuntime._run_loop)
        assert "run_streaming" not in source, (
            "_run_loop 源码不应包含 run_streaming 调用"
        )

    def test_run_loop_no_try_claim_task_call(self):
        source = inspect.getsource(TeammateRuntime._run_loop)
        assert "_try_claim_task" not in source, (
            "_run_loop 源码不应包含 _try_claim_task 调用"
        )

    def test_handle_messages_no_assignment_queue(self):
        source = inspect.getsource(TeammateRuntime._handle_messages)
        assert "_assignment_queue" not in source, (
            "_handle_messages 源码不应包含 _assignment_queue 引用"
        )

    def test_handle_messages_no_dequeue_assignment(self):
        source = inspect.getsource(TeammateRuntime._handle_messages)
        assert "_dequeue_assignment" not in source, (
            "_handle_messages 源码不应包含 _dequeue_assignment 调用"
        )


# -- 功能测试：消息路由 --


class TestMessageRouting:
    """"message" kind 消息路由到 _instruction_queue 而非触发执行。"""

    async def test_message_goes_to_instruction_queue(self, tmp_path):
        """send "message" kind → _instruction_queue 收到内容，不触发执行。"""
        rt, bus = _make_teammate(tmp_path, name="msg_rt")
        await rt.start("")
        await asyncio.sleep(0.05)

        await bus.send(LEAD_NAME, "msg_rt", "hello from lead", kind="message")
        rt.notify()
        await asyncio.sleep(0.1)

        assert not rt._instruction_queue.empty(), (
            "_instruction_queue 应收到 message 内容"
        )
        content = rt._instruction_queue.get_nowait()
        assert content == "hello from lead"

        await rt.shutdown()

    async def test_task_submitted_goes_to_instruction_queue(self, tmp_path):
        """send "task_submitted" kind → _instruction_queue 收到内容。"""
        rt, bus = _make_teammate(tmp_path, name="ts_rt")
        await rt.start("")
        await asyncio.sleep(0.05)

        await bus.send(
            LEAD_NAME, "ts_rt", "task content", kind="task_submitted", task_id="t-1"
        )
        rt.notify()
        await asyncio.sleep(0.1)

        assert not rt._instruction_queue.empty()
        content = rt._instruction_queue.get_nowait()
        assert content == "task content"

        await rt.shutdown()

    async def test_send_message_no_work_done(self, tmp_path):
        """send "message" kind → 不产生 work_done / work_failed 消息。"""
        rt, bus = _make_teammate(tmp_path, name="no_exec")
        await rt.start("")
        await asyncio.sleep(0.05)

        await bus.send(LEAD_NAME, "no_exec", "just a message", kind="message")
        rt.notify()
        await asyncio.sleep(0.3)

        msgs = await bus.consume(LEAD_NAME)
        work_kinds = {m.kind for m in msgs if m.kind.startswith("work_")}
        assert work_kinds == set(), (
            f"不应产生 work_* 消息，实际收到: {work_kinds}"
        )

        await rt.shutdown()


# -- 功能测试：build_agent_executor 保留 --


class TestBuildAgentExecutorPreserved:
    """build_agent_executor() 仍然可用，供 PersistentRuntimeV2 装配。"""

    def test_build_agent_executor_exists(self):
        assert hasattr(TeammateRuntime, "build_agent_executor")
        assert callable(TeammateRuntime.build_agent_executor)

    def test_build_agent_executor_returns_executor(self, tmp_path):
        rt, _ = _make_teammate(tmp_path)
        executor = rt.build_agent_executor()
        assert executor is not None
        assert hasattr(executor, "execute"), (
            "build_agent_executor 应返回含 execute 方法的 executor"
        )


# -- 功能测试：基本生命周期 --


class TestTeammateRuntimeLifecycle:
    """TeammateRuntime 仍可正常 start/stop（不执行任务）。"""

    async def test_start_stop(self, tmp_path):
        rt, _ = _make_teammate(tmp_path)
        await rt.start("")
        assert rt.alive
        await asyncio.sleep(0.05)
        await rt.shutdown()
        assert not rt.alive

    async def test_start_with_task_does_not_execute(self, tmp_path):
        """start("do work") 不触发 AgentLoop 执行。"""
        rt, bus = _make_teammate(tmp_path, name="no_exec_start")
        await rt.start("do work")
        assert rt.alive
        await asyncio.sleep(0.3)

        msgs = await bus.consume(LEAD_NAME)
        work_kinds = {m.kind for m in msgs if m.kind.startswith("work_")}
        assert work_kinds == set(), (
            f"start(task) 不应触发执行，实际收到: {work_kinds}"
        )

        await rt.shutdown()

    async def test_shutdown_is_fast_when_no_execution(self, tmp_path):
        """无执行时 shutdown 应快速完成。"""
        import time

        rt, _ = _make_teammate(tmp_path)
        await rt.start("")
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await rt.shutdown(timeout=5.0)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"shutdown 耗时 {elapsed:.3f}s，应快速完成"
        assert not rt.alive
