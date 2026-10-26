"""长驻 teammate：多 Agent 协作的核心运行时。

teammate 是持续运行的子 Agent，通过 MessageBus 与 lead 通信，
可自动 claim task、提交 plan 等审批、在隔离 worktree 中工作。
与一次性 subagent 不同，teammate 直到 lead 请求 shutdown 才退出。
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput
from mcc_llm import LlmProvider
from mcc_tools import ToolRegistry

from .agent import AgentLoop, model_price_per_1m
from .hooks import HookRegistry
from .mailbox import MessageBus
from .permission import PermissionChecker
from .plan import PlanStore

LEAD_NAME = "lead"


@dataclass
class TeammateSpec:
    """teammate 规格：角色、模型、工具与护栏。"""

    name: str
    role: str
    system: str
    model: str
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 20
    max_tokens: int = 4096
    budget_tokens: int = 150_000
    require_plan: bool = False
    timeout_secs: float = 300.0


@dataclass
class TeammateInfo:
    """teammate 运行时信息快照。"""

    name: str
    role: str
    status: str
    task_id: str | None = None
    worktree: str | None = None


class TeammateRuntime:
    """长驻 teammate 运行时：asyncio.Task 驱动的 WORK/IDLE 循环。

    WORK：consume inbox → 处理消息 → 有任务则执行 AgentLoop.run。
    IDLE：尝试 claim 下一个 ready task → 有则 WORK → 无则等待唤醒。
    lead 通过 ``notify`` 唤醒，通过 ``shutdown`` 请求退出。
    """

    def __init__(
        self,
        spec: TeammateSpec,
        llm: LlmProvider,
        registry: ToolRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: Any | None = None,
        base_dir: Path | None = None,
        worktree_enabled: bool = False,
        hooks: HookRegistry | None = None,
        permission: PermissionChecker | None = None,
        lead_name: str = LEAD_NAME,
    ) -> None:
        self.spec = spec
        self._llm = llm
        self._registry = registry
        self._bus = bus
        self._plan_store = plan_store
        self._task_store = task_store
        self._base_dir = base_dir
        self._worktree_enabled = worktree_enabled
        self._hooks = hooks
        self._permission = permission
        self._lead_name = lead_name

        self._task: asyncio.Task[None] | None = None
        self._shutdown_evt = asyncio.Event()
        self._wake = asyncio.Event()
        self._assignment: str | None = None
        self._task_id: str | None = None
        self._work_version = 0
        self._worktree: Path | None = None
        self._status = "idle"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def status(self) -> str:
        return self._status

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, initial_prompt: str, task_id: str | None = None) -> None:
        """启动 teammate 后台任务。"""
        self._assignment = initial_prompt
        self._task_id = task_id
        self._work_version = 0
        self._status = "working"
        self._task = asyncio.create_task(self._run_loop())

    def notify(self) -> None:
        """唤醒处于 IDLE 等待的 teammate。"""
        self._wake.set()

    async def _run_loop(self) -> None:
        try:
            while not self._shutdown_evt.is_set():
                self._wake.clear()

                inbox = await self._bus.consume(self.name)
                if inbox:
                    self._handle_messages(inbox)

                if self._shutdown_evt.is_set():
                    break

                if self._assignment:
                    self._status = "working"
                    await self._do_work()
                    self._assignment = None
                    continue

                self._status = "idle"
                if self._try_claim_task():
                    continue

                await self._wake.wait()
        except Exception as e:
            await self._bus.send(
                self.name, self._lead_name,
                f"teammate crashed: {type(e).__name__}: {e}",
                kind="error",
            )
        finally:
            self._status = "shutdown"

    def _handle_messages(self, inbox: list) -> None:
        for msg in inbox:
            if msg.kind == "shutdown":
                self._shutdown_evt.set()
            elif msg.kind == "assignment":
                self._assignment = msg.content
                tid = msg.meta.get("task_id")
                if tid:
                    self._task_id = tid
                self._work_version += 1
            elif msg.kind == "plan_approval_response":
                pass
            else:
                self._assignment = msg.content
                self._work_version += 1

    async def _do_work(self) -> None:
        if self.spec.require_plan:
            approved = await self._submit_and_wait_plan()
            if not approved:
                return

        cwd = self._prepare_worktree()

        agent = self._build_agent(cwd)
        try:
            run = await asyncio.wait_for(
                agent.run(self._assignment), timeout=self.spec.timeout_secs
            )
            summary = run.final_text or "(no output)"
        except TimeoutError:
            summary = f"teammate timed out after {self.spec.timeout_secs:.0f}s"
        except Exception as e:
            summary = f"teammate error: {type(e).__name__}: {e}"

        await self._bus.send(
            self.name, self._lead_name, summary,
            kind="work_done",
            meta={"task_id": self._task_id} if self._task_id else {},
        )

        if self._task_id and self._task_store:
            with contextlib.suppress(Exception):
                self._task_store.complete(self._task_id, owner=self.name)

        self._cleanup_worktree()

    async def _submit_and_wait_plan(self) -> bool:
        plan_req = await self._plan_store.create(
            sender=self.name,
            plan=self._assignment or "",
            task_id=self._task_id,
            work_version=self._work_version,
        )
        await self._bus.send(
            self.name, self._lead_name,
            f"Plan submitted for review:\n{self._assignment}",
            kind="plan_submitted",
            meta={"request_id": plan_req.request_id},
        )
        reviewed = await self._plan_store.wait_for_review(plan_req.request_id)
        if reviewed.status == "rejected":
            await self._bus.send(
                self.name, self._lead_name,
                f"Plan rejected: {reviewed.feedback}",
                kind="plan_rejected",
            )
            return False
        return True

    def _prepare_worktree(self) -> Path | None:
        if not self._worktree_enabled or not self._base_dir or not self._task_id:
            return None
        from .worktree import WorktreeError, create_worktree

        wt_name = f"{self.name}_{self._work_version}"
        try:
            wt = create_worktree(wt_name, self._task_id, self._base_dir)
            self._worktree = wt
            return wt
        except WorktreeError:
            return None

    def _cleanup_worktree(self) -> None:
        if not self._worktree or not self._base_dir:
            return
        from .worktree import remove_worktree

        wt_name = f"{self.name}_{self._work_version}"
        with contextlib.suppress(Exception):
            remove_worktree(wt_name, self._base_dir)
        self._worktree = None

    def _build_agent(self, cwd: Path | None = None) -> AgentLoop:
        teammate_registry = self._registry.subset(self.spec.tools)

        teammate_registry.register(
            SendToLeadTool(self._bus, self.name, self._lead_name)
        )
        if self.spec.require_plan:
            teammate_registry.register(
                SubmitPlanTool(self._plan_store, self._bus, self.name, self._lead_name)
            )

        if self._task_store:
            from mcc_tools import (
                ClaimTaskTool,
                CompleteTaskTool,
                ListTasksTool,
            )

            teammate_registry.register(ListTasksTool(self._task_store))
            teammate_registry.register(ClaimTaskTool(self._task_store))
            teammate_registry.register(CompleteTaskTool(self._task_store))

        ctx = ToolContext(
            cwd=cwd or self._base_dir or Path.cwd(),
            session_id=f"teammate::{self.name}",
            depth=1,
        )

        in_m, out_m = model_price_per_1m(self.spec.model)
        budget_usd = self.spec.budget_tokens / 1_000_000.0 * (in_m + out_m) / 2.0

        return AgentLoop(
            llm=self._llm,
            registry=teammate_registry,
            ctx=ctx,
            system=self.spec.system,
            model=self.spec.model,
            max_tokens=self.spec.max_tokens,
            max_iterations=self.spec.max_iterations,
            temperature=0.0,
            permission=self._permission,
            max_cost_usd=budget_usd,
            hooks=self._hooks,
        )

    def _try_claim_task(self) -> bool:
        if not self._task_store:
            return False
        for t in self._task_store.list():
            if t.status != "pending" or not self._task_store.can_start(t.id):
                continue
            with contextlib.suppress(Exception):
                self._task_store.claim(t.id, owner=self.name)
                self._assignment = t.description or t.subject
                self._task_id = t.id
                self._work_version += 1
                return True
        return False

    async def shutdown(self) -> None:
        """请求 teammate 关闭并等待退出。"""
        self._shutdown_evt.set()
        self.notify()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    def info(self) -> TeammateInfo:
        return TeammateInfo(
            name=self.name,
            role=self.spec.role,
            status=self._status,
            task_id=self._task_id,
            worktree=str(self._worktree) if self._worktree else None,
        )


class TeammateManager:
    """teammate 生命周期管理：spawn / list / message / shutdown / lead inbox。

    持有 MessageBus、PlanStore 与所有活跃 TeammateRuntime，
    为 lead 工具提供统一入口。
    """

    def __init__(
        self,
        llm: LlmProvider,
        registry: ToolRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: Any | None = None,
        base_dir: Path | None = None,
        worktree_enabled: bool = False,
        max_teammates: int = 4,
        default_model: str = "claude-haiku-4-5",
        hooks: HookRegistry | None = None,
        permission: PermissionChecker | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._bus = bus
        self._plan_store = plan_store
        self._task_store = task_store
        self._base_dir = base_dir
        self._worktree_enabled = worktree_enabled
        self._max_teammates = max_teammates
        self._default_model = default_model
        self._hooks = hooks
        self._permission = permission
        self._teammates: dict[str, TeammateRuntime] = {}

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def plan_store(self) -> PlanStore:
        return self._plan_store

    def list_active(self) -> list[TeammateInfo]:
        return [rt.info() for rt in self._teammates.values() if rt.alive]

    async def spawn(
        self,
        name: str,
        role: str,
        prompt: str,
        model: str | None = None,
        task_id: str | None = None,
        require_plan: bool = False,
        tools: list[str] | None = None,
    ) -> str:
        """启动一个长驻 teammate，返回其名称。"""
        if name in self._teammates and self._teammates[name].alive:
            raise ValueError(f"teammate already active: {name}")
        if len(self.list_active()) >= self._max_teammates:
            raise ValueError(f"max teammates limit reached: {self._max_teammates}")

        spec = TeammateSpec(
            name=name,
            role=role,
            system=_teammate_system(role),
            model=model or self._default_model,
            tools=tools or _default_teammate_tools(),
            require_plan=require_plan,
        )
        rt = TeammateRuntime(
            spec=spec,
            llm=self._llm,
            registry=self._registry,
            bus=self._bus,
            plan_store=self._plan_store,
            task_store=self._task_store,
            base_dir=self._base_dir,
            worktree_enabled=self._worktree_enabled,
            hooks=self._hooks,
            permission=self._permission,
        )
        self._teammates[name] = rt
        await rt.start(prompt, task_id=task_id)
        return name

    async def send_message(self, to: str, content: str) -> str:
        """lead 向 teammate 发消息并唤醒。"""
        if to not in self._teammates or not self._teammates[to].alive:
            raise ValueError(f"no active teammate: {to}")
        await self._bus.send(LEAD_NAME, to, content, kind="message")
        self._teammates[to].notify()
        return f"Message sent to {to}"

    async def request_shutdown(self, name: str) -> str:
        """请求 teammate 关闭。"""
        rt = self._teammates.get(name)
        if rt is None or not rt.alive:
            raise ValueError(f"no active teammate: {name}")
        await self._bus.send(LEAD_NAME, name, "shutdown", kind="shutdown")
        rt.notify()
        await rt.shutdown()
        return f"Teammate {name} shut down"

    async def shutdown_all(self) -> None:
        """关闭所有活跃 teammate。"""
        names = [n for n, rt in self._teammates.items() if rt.alive]
        for name in names:
            with contextlib.suppress(Exception):
                await self.request_shutdown(name)

    async def consume_lead_inbox(self) -> list:
        """读取 lead 的全部未读消息。"""
        return await self._bus.consume(LEAD_NAME)

    async def peek_lead_inbox(self) -> bool:
        return await self._bus.peek(LEAD_NAME)


def _teammate_system(role: str) -> str:
    return (
        f"你是 {role}，一个长驻协作 Agent。\n"
        "通过 send_to_lead 向 lead 汇报进展和结果。\n"
        "用 list_tasks 查看可用任务，claim_task 领取，complete_task 完成。\n"
        "专注于自己的任务，不要越界操作其他 teammate 的工作。\n"
        "完成后用 send_to_lead 发送简短摘要。"
    )


def _default_teammate_tools() -> list[str]:
    return [
        "read_file", "list_dir", "write_file", "edit_file",
        "run_bash", "grep", "glob",
    ]


def format_team_events(messages: list) -> str:
    """将 lead inbox 消息格式化为注入 prompt。"""
    parts = []
    for msg in messages:
        tag = msg.kind.upper()
        parts.append(f"[{tag} from {msg.sender}] {msg.content}")
    return "[team events]\n" + "\n".join(parts)


# -- Teammate 专属工具 --


class SendToLeadTool:
    """teammate 向 lead 发送消息。"""

    def __init__(self, bus: MessageBus, sender: str, lead_name: str = LEAD_NAME) -> None:
        self._bus = bus
        self._sender = sender
        self._lead = lead_name

    @property
    def name(self) -> str:
        return "send_to_lead"

    @property
    def description(self) -> str:
        return "Send a message to the lead agent (progress, questions, results)."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Message content for the lead",
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        content = str(input.get("content", ""))
        if not content:
            return ToolOutput.err("'content' is required")
        await self._bus.send(self._sender, self._lead, content, kind="message")
        return ToolOutput.ok("Message sent to lead")


class SubmitPlanTool:
    """teammate 提交工作计划等待 lead 审批。"""

    def __init__(
        self,
        plan_store: PlanStore,
        bus: MessageBus,
        sender: str,
        lead_name: str = LEAD_NAME,
    ) -> None:
        self._plan_store = plan_store
        self._bus = bus
        self._sender = sender
        self._lead = lead_name

    @property
    def name(self) -> str:
        return "submit_plan"

    @property
    def description(self) -> str:
        return "Submit a work plan for lead approval before making changes."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["plan"],
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Detailed work plan for approval",
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        plan = str(input.get("plan", ""))
        if not plan:
            return ToolOutput.err("'plan' is required")
        req = await self._plan_store.create(sender=self._sender, plan=plan)
        await self._bus.send(
            self._sender, self._lead,
            f"Plan submitted for review:\n{plan}",
            kind="plan_submitted",
            meta={"request_id": req.request_id},
        )
        reviewed = await self._plan_store.wait_for_review(req.request_id)
        if reviewed.status == "approved":
            return ToolOutput.ok(
                f"Plan approved. {reviewed.feedback}".strip()
            )
        return ToolOutput.err(
            f"Plan rejected. {reviewed.feedback}".strip()
        )


# -- Lead 专属工具 --


class SpawnTeammateTool:
    """spawn_teammate：lead 启动一个长驻 teammate。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "spawn_teammate"

    @property
    def description(self) -> str:
        return (
            "Spawn a persistent teammate that runs until shutdown. "
            "The teammate works on the given prompt and reports back via messages."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["name", "role", "prompt"],
            "properties": {
                "name": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,64}$"},
                "role": {"type": "string"},
                "prompt": {"type": "string"},
                "model": {"type": "string"},
                "task_id": {"type": "string"},
                "require_plan": {"type": "boolean"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        name = str(input.get("name", ""))
        role = str(input.get("role", ""))
        prompt = str(input.get("prompt", ""))
        if not name or not role or not prompt:
            return ToolOutput.err("'name', 'role', 'prompt' are all required")
        try:
            await self._manager.spawn(
                name=name,
                role=role,
                prompt=prompt,
                model=input.get("model"),
                task_id=input.get("task_id"),
                require_plan=bool(input.get("require_plan", False)),
            )
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(f"Teammate '{name}' spawned with role: {role}")


class ListTeammatesTool:
    """list_teammates：列出所有活跃 teammate。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "list_teammates"

    @property
    def description(self) -> str:
        return "List all active teammates with their status."

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        infos = self._manager.list_active()
        if not infos:
            return ToolOutput.ok("No active teammates")
        lines = []
        for info in infos:
            line = f"- {info.name} ({info.role}): {info.status}"
            if info.task_id:
                line += f" task={info.task_id}"
            if info.worktree:
                line += f" wt={info.worktree}"
            lines.append(line)
        return ToolOutput.ok("\n".join(lines))


class SendTeammateMessageTool:
    """send_message：lead 向 teammate 发消息。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "send_message"

    @property
    def description(self) -> str:
        return "Send a message to an active teammate."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["to", "content"],
            "properties": {
                "to": {"type": "string"},
                "content": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        to = str(input.get("to", ""))
        content = str(input.get("content", ""))
        if not to or not content:
            return ToolOutput.err("'to' and 'content' are both required")
        try:
            result = await self._manager.send_message(to, content)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class RequestShutdownTool:
    """request_shutdown：lead 请求 teammate 关闭。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "request_shutdown"

    @property
    def description(self) -> str:
        return "Ask a teammate to shut down gracefully."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["teammate"],
            "properties": {
                "teammate": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        teammate = str(input.get("teammate", ""))
        if not teammate:
            return ToolOutput.err("'teammate' is required")
        try:
            result = await self._manager.request_shutdown(teammate)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class ReviewPlanTool:
    """review_plan：lead 审批或驳回 teammate 的 plan。"""

    def __init__(self, plan_store: PlanStore, bus: MessageBus) -> None:
        self._plan_store = plan_store
        self._bus = bus

    @property
    def name(self) -> str:
        return "review_plan"

    @property
    def description(self) -> str:
        return "Approve or reject a teammate's submitted plan."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["request_id", "approve"],
            "properties": {
                "request_id": {"type": "string"},
                "approve": {"type": "boolean"},
                "feedback": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        rid = str(input.get("request_id", ""))
        approve = bool(input.get("approve", False))
        feedback = str(input.get("feedback", ""))
        if not rid:
            return ToolOutput.err("'request_id' is required")
        try:
            req = await self._plan_store.review(rid, approve, feedback)
        except Exception as e:
            return ToolOutput.err(str(e))
        await self._bus.send(
            LEAD_NAME, req.sender,
            feedback or ("Plan approved." if approve else "Revise and resubmit."),
            kind="plan_approval_response",
            meta={"request_id": rid, "approve": approve},
        )
        return ToolOutput.ok(f"Plan {req.status} ({rid})")


class CreateWorktreeTool:
    """create_worktree：lead 为 task 创建隔离工作目录。"""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    @property
    def name(self) -> str:
        return "create_worktree"

    @property
    def description(self) -> str:
        return "Create an isolated git worktree bound to a task."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["name", "task_id"],
            "properties": {
                "name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"},
                "task_id": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        from .worktree import WorktreeError, create_worktree

        name = str(input.get("name", ""))
        task_id = str(input.get("task_id", ""))
        if not name or not task_id:
            return ToolOutput.err("'name' and 'task_id' are both required")
        try:
            wt = create_worktree(name, task_id, self._base_dir)
        except WorktreeError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(f"Worktree created at {wt}")


def register_team_tools(registry: ToolRegistry, manager: TeammateManager) -> None:
    """将所有 lead 团队工具注册到 registry。"""
    registry.register(SpawnTeammateTool(manager))
    registry.register(ListTeammatesTool(manager))
    registry.register(SendTeammateMessageTool(manager))
    registry.register(RequestShutdownTool(manager))
    registry.register(ReviewPlanTool(manager.plan_store, manager.bus))


__all__ = [
    "CreateWorktreeTool",
    "LEAD_NAME",
    "ListTeammatesTool",
    "MessageBus",
    "PlanStore",
    "RequestShutdownTool",
    "ReviewPlanTool",
    "SendToLeadTool",
    "SendTeammateMessageTool",
    "SpawnTeammateTool",
    "SubmitPlanTool",
    "TeammateInfo",
    "TeammateManager",
    "TeammateRuntime",
    "TeammateSpec",
    "format_team_events",
    "register_team_tools",
]
