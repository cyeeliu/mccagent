"""AgentApplication 便捷方法与生命周期测试。

验证 AgentApplication 正确委托 RuntimeCoordinator，
提供统一的 Runtime 注册、查询、健康检查和关闭入口。
"""

from __future__ import annotations

import asyncio

import pytest
from mcc_cli.application import AgentApplication
from mcc_core.runtime import RuntimeStatus
from mcc_harness import InteractiveRuntimeV2, RuntimeCoordinator
from mcc_llm.mock import MockLlm, single_text_turn


class FakeAgent:
    """最小 Agent 桩。"""

    async def run(self, messages, *, tools=None, system=None, model=None):
        yield


class TestAgentApplicationConvenience:
    """AgentApplication 便捷委托方法。"""

    def test_default_coordinator_created(self):
        app = AgentApplication()
        assert app.coordinator is not None
        assert isinstance(app.coordinator, RuntimeCoordinator)

    def test_register_runtime_delegates_to_coordinator(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        app.register_runtime(rt)
        assert app.coordinator.get("r1") is rt

    def test_register_runtime_with_priority(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        app.register_runtime(rt, priority=5)
        assert app.coordinator.get("r1") is rt

    async def test_register_and_start_delegates(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        await app.register_and_start(rt)
        assert rt.status == RuntimeStatus.READY
        await rt.stop()

    async def test_register_and_start_rollback_on_failure(self):
        from tests.test_coordinator_lifecycle import FailingStartRuntime

        app = AgentApplication()
        rt = FailingStartRuntime()
        with pytest.raises(RuntimeError, match="start failed"):
            await app.register_and_start(rt)
        assert app.coordinator.get("fail-1") is None

    def test_get_runtime_by_id(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        app.register_runtime(rt)
        assert app.get_runtime("r1") is rt
        assert app.get_runtime("no-such") is None

    def test_get_runtime_by_agent(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(
            FakeAgent(), runtime_id="r1", agent_id="agent-x",
        )
        app.register_runtime(rt)
        assert app.get_runtime_by_agent("agent-x") is rt
        assert app.get_runtime_by_agent("no-such") is None

    def test_list_runtimes(self):
        app = AgentApplication()
        rt1 = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        rt2 = InteractiveRuntimeV2(FakeAgent(), runtime_id="r2", agent_id="a2")
        app.register_runtime(rt1)
        app.register_runtime(rt2)
        runtimes = app.list_runtimes()
        assert len(runtimes) == 2
        assert rt1 in runtimes
        assert rt2 in runtimes

    def test_list_active_runtimes(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        app.register_runtime(rt)
        active = app.list_active_runtimes()
        assert len(active) == 0

    def test_snapshot(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        app.register_runtime(rt)
        snaps = app.snapshot()
        assert len(snaps) == 1
        assert snaps[0].runtime_id == "r1"

    async def test_health_check_all(self):
        app = AgentApplication()
        result = await app.health_check_all()
        assert result == {}


class TestAgentApplicationLifecycle:
    """AgentApplication async context manager 与 shutdown。"""

    async def test_async_context_manager(self):
        async with AgentApplication() as app:
            assert app.coordinator is not None
        assert app._shutdown is True

    async def test_shutdown_is_idempotent(self):
        app = AgentApplication()
        await app.shutdown()
        assert app._shutdown is True
        await app.shutdown()

    async def test_shutdown_stops_all_runtimes(self):
        app = AgentApplication()
        rt = InteractiveRuntimeV2(FakeAgent(), runtime_id="r1", agent_id="a1")
        await app.register_and_start(rt)
        await app.shutdown()
        assert not rt.alive

    async def test_shutdown_with_mcp_manager(self):
        class StubMcpManager:
            def __init__(self) -> None:
                self.closed = False

            async def close_all(self) -> None:
                self.closed = True

        mcp = StubMcpManager()
        app = AgentApplication(mcp_manager=mcp)
        await app.shutdown()
        assert mcp.closed is True

    async def test_shutdown_mcp_failure_does_not_block(self):
        class FailingMcpManager:
            async def close_all(self) -> None:
                raise RuntimeError("mcp close failed")

        app = AgentApplication(mcp_manager=FailingMcpManager())
        await app.shutdown()
        assert app._shutdown is True


class TestAgentApplicationTeam:
    """AgentApplication 团队操作委托方法。"""

    def test_team_enabled_false_without_manager(self):
        app = AgentApplication()
        assert not app.team_enabled

    def test_team_enabled_false_without_orchestrator(self):
        app = AgentApplication(team_manager=object())
        assert not app.team_enabled

    def test_team_enabled_true_with_both(self):
        app = AgentApplication(team_manager=object(), orchestrator=object())
        assert app.team_enabled

    def test_list_active_teammates_empty_without_manager(self):
        app = AgentApplication()
        assert app.list_active_teammates() == []

    def test_list_active_teammates_delegates(self):
        class StubManager:
            def list_active(self):
                return [("worker-1", "coder", "running")]

        app = AgentApplication(team_manager=StubManager())
        result = app.list_active_teammates()
        assert result == [("worker-1", "coder", "running")]

    def test_format_team_status_no_teammates(self):
        class StubManager:
            def list_active(self):
                return []

        app = AgentApplication(team_manager=StubManager(), orchestrator=object())
        assert "活跃 teammate: 0" in app.format_team_status()

    def test_format_team_status_with_teammates(self):
        class StubInfo:
            def __init__(self, name, role, status):
                self.name = name
                self.role = role
                self.status = status

        class StubManager:
            def list_active(self):
                return [
                    StubInfo("worker-1", "coder", "running"),
                    StubInfo("worker-2", "tester", "idle"),
                ]

        app = AgentApplication(team_manager=StubManager(), orchestrator=object())
        text = app.format_team_status()
        assert "活跃 teammate: 2" in text
        assert "worker-1" in text
        assert "worker-2" in text

    async def test_consume_lead_inbox_empty_without_manager(self):
        app = AgentApplication()
        assert await app.consume_lead_inbox() == []

    async def test_consume_lead_inbox_delegates(self):
        class StubManager:
            async def consume_lead_inbox(self):
                return ["msg-1", "msg-2"]

        app = AgentApplication(team_manager=StubManager())
        result = await app.consume_lead_inbox()
        assert result == ["msg-1", "msg-2"]

    async def test_peek_lead_inbox_false_without_manager(self):
        app = AgentApplication()
        assert await app.peek_lead_inbox() is False

    async def test_peek_lead_inbox_delegates(self):
        class StubManager:
            async def peek_lead_inbox(self):
                return True

        app = AgentApplication(team_manager=StubManager())
        assert await app.peek_lead_inbox() is True

    async def test_send_teammate_message_without_manager(self):
        app = AgentApplication()
        result = await app.send_teammate_message("worker", "hello")
        assert "未启用" in result

    async def test_send_teammate_message_delegates(self):
        class StubManager:
            async def send_message(self, to, content):
                return f"sent to {to}: {content}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.send_teammate_message("worker", "hello")
        assert "sent to worker: hello" in result

    async def test_send_instruction_delegates(self):
        class StubManager:
            async def send_instruction(self, to, content):
                return f"instruction to {to}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.send_instruction("worker", "stop")
        assert "instruction to worker" in result

    async def test_assign_task_delegates(self):
        class StubManager:
            async def assign_task(self, to, content):
                return f"task to {to}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.assign_task("worker", "do work")
        assert "task to worker" in result

    async def test_cancel_teammate_delegates(self):
        class StubManager:
            async def cancel_teammate(self, name):
                return f"cancelled {name}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.cancel_teammate("worker")
        assert "cancelled worker" in result

    async def test_pause_teammate_delegates(self):
        class StubManager:
            async def pause_teammate(self, name):
                return f"paused {name}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.pause_teammate("worker")
        assert "paused worker" in result

    async def test_resume_teammate_delegates(self):
        class StubManager:
            async def resume_teammate(self, name):
                return f"resumed {name}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.resume_teammate("worker")
        assert "resumed worker" in result

    async def test_request_shutdown_teammate_delegates(self):
        class StubManager:
            async def request_shutdown(self, name):
                return f"shutdown {name}"

        app = AgentApplication(team_manager=StubManager())
        result = await app.request_shutdown_teammate("worker")
        assert "shutdown worker" in result

    async def test_run_team_disabled_without_manager(self):
        app = AgentApplication()
        result = await app.run_team("do something")
        assert "未启用" in result


class TestAgentApplicationTeamIntegration:
    """AgentApplication 团队编排集成测试。"""

    async def test_run_team_full_flow(self, tmp_path):
        from mcc_harness import (
            AgentRegistry,
            MessageBus,
            PlanStore,
            RuntimeCoordinator,
            TeammateManager,
            TeamOrchestrator,
        )
        from mcc_tools import ToolRegistry

        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        coord = RuntimeCoordinator()
        agent_reg = AgentRegistry()
        orch = TeamOrchestrator(
            registry=agent_reg, bus=bus, plan_store=plan_store,
            coordinator=coord,
        )
        mgr = TeammateManager(
            llm=MockLlm([
                single_text_turn("initial done"),
                single_text_turn("task completed"),
            ]),
            registry=ToolRegistry(), bus=bus, plan_store=plan_store,
            base_dir=tmp_path, coordinator=coord,
            agent_registry=agent_reg, orchestrator=orch,
        )
        app = AgentApplication(
            coordinator=coord,
            team_manager=mgr,
            orchestrator=orch,
        )
        try:
            result = await app.run_team("do feature work")
            assert "TeamRun" in result
            assert "完成" in result
        finally:
            await mgr.shutdown_all()
