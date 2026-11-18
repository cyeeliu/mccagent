"""应用生命周期管理：统一 shutdown 清理与团队操作入口。

AgentApplication 持有 RuntimeCoordinator、可选的 MCP manager、
TeamOrchestrator 和 TeammateManager，在应用退出时确保所有 Runtime
和子进程干净关闭，无遗留任务。可作为 async context manager 使用，
Headless 与 TUI 共用同一实例。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentApplication:
    """应用生命周期管理器与团队操作入口。

    持有 RuntimeCoordinator，提供统一 shutdown 入口。
    持有 TeamOrchestrator 和 TeammateManager，提供团队编排操作。
    可作为 async context manager 使用。
    """

    def __init__(
        self,
        coordinator: Any | None = None,
        *,
        mcp_manager: Any | None = None,
        event_source: Any | None = None,
        team_manager: Any | None = None,
        orchestrator: Any | None = None,
    ) -> None:
        if coordinator is None:
            from mcc_harness import RuntimeCoordinator

            coordinator = RuntimeCoordinator()
        self._coordinator = coordinator
        self._mcp_manager = mcp_manager
        self._event_source = event_source
        self._team_manager = team_manager
        self._orchestrator = orchestrator
        self._shutdown = False

    @property
    def coordinator(self) -> Any | None:
        return self._coordinator

    @property
    def event_source(self) -> Any | None:
        return self._event_source

    @property
    def team_manager(self) -> Any | None:
        return self._team_manager

    @property
    def orchestrator(self) -> Any | None:
        return self._orchestrator

    @property
    def team_enabled(self) -> bool:
        """团队功能是否已启用。"""
        return self._team_manager is not None and self._orchestrator is not None

    def register_runtime(self, runtime: Any, *, priority: int = 0) -> None:
        """注册 Runtime 到 Coordinator。"""
        self._coordinator.register(runtime, priority=priority)

    async def register_and_start(
        self,
        runtime: Any,
        *,
        priority: int = 0,
    ) -> None:
        """注册并启动 Runtime，失败时自动回滚。"""
        await self._coordinator.register_and_start(runtime, priority=priority)

    def get_runtime(self, runtime_id: str) -> Any | None:
        """按 runtime_id 查询 Runtime。"""
        return self._coordinator.get(runtime_id)

    def get_runtime_by_agent(self, agent_id: str) -> Any | None:
        """按 agent_id 查询 Runtime。"""
        return self._coordinator.get_by_agent(agent_id)

    def list_runtimes(self) -> list[Any]:
        """返回所有已注册的 Runtime。"""
        return self._coordinator.list_all()

    def list_active_runtimes(self) -> list[Any]:
        """返回所有活跃的 Runtime。"""
        return self._coordinator.list_active()

    def snapshot(self) -> list[Any]:
        """返回所有 Runtime 的状态快照。"""
        return self._coordinator.snapshot()

    async def health_check_all(self) -> dict[str, Any]:
        """对所有 Runtime 执行健康检查。"""
        return await self._coordinator.health_check_all()

    async def shutdown(self, *, timeout: float = 10.0) -> None:
        """完整 8 步注销事务，幂等，聚合异常不抛出。

        顺序遵循方案 §8.3 注销事务：
        1. 标记 STOPPING — 阻止新注册与新任务
        2. 优雅关闭 TeammateManager — 通过消息总线发送 shutdown（需 runtime 存活）
        3. 取消所有运行中任务 — 向活跃 Runtime 发送 cancel 信号
        4. 停止所有 Runtime — 含停止健康/退出监控，按优先级停止
        5. 关闭 MCP 子进程
        6. 关闭 EventSource — 关闭事件管道
        7. 清理终端 — 移除已到达终态的 Runtime
        8. 汇总报告 — 记录失败与超时
        """
        if self._shutdown:
            return
        self._shutdown = True

        # Step 2: 优雅关闭 TeammateManager（需 runtime 存活才能通过消息总线通信）
        if self._team_manager is not None:
            try:
                await self._team_manager.shutdown_all()
            except Exception:
                logger.warning(
                    "shutdown step 2: team_manager.shutdown_all failed",
                    exc_info=True,
                )

        # Step 3: 取消所有运行中任务
        try:
            await self._coordinator.cancel_all()
        except Exception:
            logger.warning(
                "shutdown step 3: coordinator.cancel_all failed",
                exc_info=True,
            )

        # Step 4: 停止所有 Runtime（含停止健康/退出监控）
        report = await self._coordinator.stop_all(timeout=timeout)

        # Step 5: 关闭 MCP 子进程
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.close_all()
            except Exception:
                logger.warning(
                    "shutdown step 5: mcp_manager.close_all failed",
                    exc_info=True,
                )

        # Step 6: 关闭 EventSource
        if self._event_source is not None:
            try:
                await self._event_source.shutdown(timeout=timeout)
            except Exception:
                logger.warning(
                    "shutdown step 6: event_source.shutdown failed",
                    exc_info=True,
                )

        # Step 7: 清理终端 Runtime
        try:
            self._coordinator.clear_terminal()
        except Exception:
            logger.warning(
                "shutdown step 7: coordinator.clear_terminal failed",
                exc_info=True,
            )

        # Step 8: 汇总报告
        if report.has_errors:
            logger.warning(
                "shutdown completed with %d failure(s), %d timeout(s): %s",
                len(report.failed),
                len(report.timed_out),
                {"failed": report.failed, "timed_out": report.timed_out},
            )

    # -- 团队操作 --

    def list_active_teammates(self) -> list[Any]:
        """返回所有活跃 teammate 的信息列表。"""
        if self._team_manager is None:
            return []
        return self._team_manager.list_active()

    def format_team_status(self) -> str:
        """格式化当前团队状态文本。"""
        active = self.list_active_teammates()
        lines = [f"活跃 teammate: {len(active)}"]
        for info in active:
            lines.append(f"  {info.name} ({info.role}) — {info.status}")
        return "\n".join(lines)

    async def run_team(self, objective: str) -> str:
        """触发完整团队编排闭环：plan → spawn → start → execute → collect。

        返回格式化的结果文本。
        """
        if not self.team_enabled:
            return "团队功能未启用（config.team.enabled=false）"

        team = self._orchestrator.create_team("default", "default", "lead")
        specs = await self._orchestrator.planner.plan(objective)
        active_names = {info.name for info in self._team_manager.list_active()}
        for i in range(len(specs)):
            worker_name = f"worker-{i + 1}"
            if worker_name not in active_names:
                await self._team_manager.spawn(
                    name=worker_name,
                    role="worker",
                    prompt="",
                )

        run = await self._orchestrator.start(team, objective)
        if not run.assignments:
            return f"TeamRun {run.run_id} 已启动但无任务分配（planner 返回空）"

        await self._orchestrator.execute_run(run.run_id)
        result = await self._orchestrator.collect_results(run.run_id)

        lines = [
            f"TeamRun {result.run_id} — {result.status.value}",
            f"完成: {result.completed_count}/{result.total_count}, "
            f"失败: {result.failed_count}",
        ]
        for r in result.results:
            marker = "✓" if r.status == "completed" else "✗"
            lines.append(
                f"  {marker} {r.agent_id}: {r.summary or r.error or ''}"
            )
        return "\n".join(lines)

    async def consume_lead_inbox(self) -> list[Any]:
        """消费 lead inbox 中的待处理消息。"""
        if self._team_manager is None:
            return []
        return await self._team_manager.consume_lead_inbox()

    async def peek_lead_inbox(self) -> bool:
        """检查 lead inbox 是否有待处理消息。"""
        if self._team_manager is None:
            return False
        return await self._team_manager.peek_lead_inbox()

    async def send_teammate_message(self, to: str, content: str) -> str:
        """向指定 teammate 发送消息。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.send_message(to, content)

    async def send_instruction(self, to: str, content: str) -> str:
        """向指定 teammate 注入运行时指令。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.send_instruction(to, content)

    async def assign_task(self, to: str, content: str) -> str:
        """向指定 teammate 分配新任务。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.assign_task(to, content)

    async def cancel_teammate(self, name: str) -> str:
        """取消指定 teammate 的当前任务。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.cancel_teammate(name)

    async def pause_teammate(self, name: str) -> str:
        """暂停指定 teammate。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.pause_teammate(name)

    async def resume_teammate(self, name: str) -> str:
        """恢复指定 teammate。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.resume_teammate(name)

    async def request_shutdown_teammate(self, name: str) -> str:
        """请求关闭指定 teammate。"""
        if self._team_manager is None:
            return "团队功能未启用"
        return await self._team_manager.request_shutdown(name)

    async def __aenter__(self) -> AgentApplication:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.shutdown()
