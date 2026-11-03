"""长驻 teammate：多 Agent 协作的核心运行时。

teammate 是持续运行的子 Agent，通过 MessageBus 与 lead 通信，
可自动 claim task、提交 plan 等审批、在隔离 worktree 中工作。
与一次性 subagent 不同，teammate 直到 lead 请求 shutdown 才退出。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
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


class TeammateStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    IDLE = "idle"
    ASSIGNED = "assigned"
    PLANNING = "planning"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    REPORTING = "reporting"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RESTARTING = "restarting"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


_TEAMMATE_TRANSITIONS: dict[TeammateStatus, frozenset[TeammateStatus]] = {
    TeammateStatus.CREATED: frozenset({
        TeammateStatus.STARTING,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.STARTING: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.ASSIGNED,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.IDLE: frozenset({
        TeammateStatus.ASSIGNED,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.ASSIGNED: frozenset({
        TeammateStatus.PLANNING,
        TeammateStatus.RUNNING,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.PLANNING: frozenset({
        TeammateStatus.WAITING_APPROVAL,
        TeammateStatus.FAILED,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.WAITING_APPROVAL: frozenset({
        TeammateStatus.RUNNING,
        TeammateStatus.IDLE,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.RUNNING: frozenset({
        TeammateStatus.REPORTING,
        TeammateStatus.FAILED,
        TeammateStatus.TIMEOUT,
        TeammateStatus.CANCELLED,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.REPORTING: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.FAILED: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.RESTARTING,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.TIMEOUT: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.RESTARTING,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.CANCELLED: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.RESTARTING: frozenset({
        TeammateStatus.IDLE,
        TeammateStatus.SHUTTING_DOWN,
    }),
    TeammateStatus.SHUTTING_DOWN: frozenset({TeammateStatus.SHUTDOWN}),
    TeammateStatus.SHUTDOWN: frozenset(),
}


class TeammateStateError(RuntimeError):
    """teammate 状态迁移非法。"""


@dataclass
class TransitionRecord:
    """单次状态迁移记录：来源、目标、原因与时间戳。"""

    from_status: TeammateStatus
    to_status: TeammateStatus
    reason: str
    timestamp: float


@dataclass
class CurrentExecution:
    """当前执行上下文：任务终态后必须清空。"""

    task_id: str | None = None
    assignment: str | None = None
    work_version: int = 0
    started_at: float | None = None
    cancel_token: asyncio.Event | None = None

    def clear(self) -> None:
        self.task_id = None
        self.assignment = None
        self.work_version = 0
        self.started_at = None
        self.cancel_token = None

    @property
    def active(self) -> bool:
        return self.task_id is not None or self.assignment is not None


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
    last_transition: str | None = None


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
        self._current = CurrentExecution()
        self._worktree: Path | None = None
        self._status = TeammateStatus.CREATED
        self._last_transition: TransitionRecord | None = None
        self._force_stop = False
        self._assignment_queue: list[dict[str, Any]] = []
        self._instruction_queue: asyncio.Queue[str] = asyncio.Queue()
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def status(self) -> str:
        return self._status.value

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def _transition(self, target: TeammateStatus, reason: str = "") -> None:
        current = self._status
        allowed = _TEAMMATE_TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise TeammateStateError(
                f"teammate {self.name} cannot transition from "
                f"{current.value} to {target.value}"
            )
        self._status = target
        self._last_transition = TransitionRecord(
            from_status=current,
            to_status=target,
            reason=reason,
            timestamp=time.monotonic(),
        )

    def _safe_shutdown(self, reason: str = "") -> None:
        if self._status == TeammateStatus.SHUTDOWN:
            return
        if self._status != TeammateStatus.SHUTTING_DOWN:
            self._transition(
                TeammateStatus.SHUTTING_DOWN,
                reason=reason or ("force stop" if self._force_stop else "graceful"),
            )
        self._transition(
            TeammateStatus.SHUTDOWN,
            reason="force terminated" if self._force_stop else "graceful exit",
        )

    async def start(self, initial_prompt: str, task_id: str | None = None) -> None:
        """启动 teammate 后台任务。"""
        self._transition(TeammateStatus.STARTING, reason="start")
        self._current.assignment = initial_prompt
        self._current.task_id = task_id
        self._current.work_version = 0
        self._current.started_at = time.monotonic()
        self._task = asyncio.create_task(self._run_loop())

    def notify(self) -> None:
        """唤醒处于 IDLE 等待的 teammate。"""
        self._wake.set()

    async def _run_loop(self) -> None:
        try:
            if self._current.assignment:
                self._transition(
                    TeammateStatus.ASSIGNED, reason="initial assignment"
                )
            else:
                self._transition(TeammateStatus.IDLE, reason="no initial work")

            while not self._shutdown_evt.is_set():
                self._wake.clear()

                inbox = await self._bus.consume(self.name)
                if inbox:
                    self._handle_messages(inbox)

                if self._shutdown_evt.is_set():
                    break

                if self._current.assignment:
                    if self._status == TeammateStatus.IDLE:
                        self._transition(
                            TeammateStatus.ASSIGNED, reason="assignment ready"
                        )
                    await self._do_work()
                    self._current.clear()
                    if self._dequeue_assignment():
                        continue
                    if self._status != TeammateStatus.IDLE:
                        self._transition(
                            TeammateStatus.IDLE, reason="work done"
                        )
                    continue

                if self._try_claim_task():
                    continue

                if self._status != TeammateStatus.IDLE:
                    self._transition(TeammateStatus.IDLE, reason="idle")
                await self._wake.wait()
        except Exception as e:
            await self._bus.send(
                self.name, self._lead_name,
                f"teammate crashed: {type(e).__name__}: {e}",
                kind="error",
            )
        finally:
            self._safe_shutdown(reason="loop exit")

    def _dequeue_assignment(self) -> bool:
        """从任务队列取出下一个 assignment 到 _current。"""
        if not self._assignment_queue:
            return False
        item = self._assignment_queue.pop(0)
        self._current.assignment = item["content"]
        self._current.task_id = item.get("task_id")
        self._current.work_version += 1
        self._current.started_at = time.monotonic()
        return True

    def _handle_messages(self, inbox: list) -> None:
        for msg in inbox:
            if msg.kind == "shutdown":
                self._shutdown_evt.set()
            elif msg.kind in ("assignment", "message"):
                self._assignment_queue.append({
                    "content": msg.content,
                    "task_id": msg.meta.get("task_id") if msg.meta else None,
                })
                if not self._current.assignment:
                    self._dequeue_assignment()
            elif msg.kind == "instruction":
                self._instruction_queue.put_nowait(msg.content)
            elif msg.kind == "cancel":
                if self._current.cancel_token is not None:
                    self._current.cancel_token.set()
            elif msg.kind == "pause":
                self._pause_event.clear()
            elif msg.kind == "resume":
                self._pause_event.set()
                self._wake.set()
            elif msg.kind == "plan_approval_response":
                pass
            else:
                self._instruction_queue.put_nowait(msg.content)

    async def _do_work(self) -> None:
        if self.spec.require_plan:
            self._transition(TeammateStatus.PLANNING, reason="require_plan")
            approved = await self._submit_and_wait_plan()
            if not approved:
                self._transition(TeammateStatus.IDLE, reason="plan rejected")
                self._current.clear()
                return
            self._transition(TeammateStatus.RUNNING, reason="plan approved")
        else:
            self._transition(TeammateStatus.RUNNING, reason="start work")

        from .worktree import WorktreeError as _WtErr

        try:
            cwd = self._prepare_worktree()
        except _WtErr as e:
            summary = f"worktree creation failed: {e}"
            self._transition(TeammateStatus.FAILED, reason="worktree creation failed")
            await self._bus.send(
                self.name, self._lead_name, summary,
                kind="work_failed",
                meta={"task_id": self._current.task_id} if self._current.task_id else {},
            )
            if self._current.task_id and self._task_store:
                with contextlib.suppress(Exception):
                    self._task_store.fail(
                        self._current.task_id,
                        error=summary,
                        owner=self.name,
                    )
            self._transition(
                TeammateStatus.IDLE, reason="recover from failed"
            )
            self._current.clear()
            return

        agent = self._build_agent(cwd)
        self._current.cancel_token = asyncio.Event()
        outcome: TeammateStatus
        try:
            from .agent import AgentEvent as _Ev
            queue: asyncio.Queue[_Ev] = asyncio.Queue()
            run = await asyncio.wait_for(
                agent.run_streaming(
                    self._current.assignment,
                    queue,
                    instruction_queue=self._instruction_queue,
                    cancel_event=self._current.cancel_token,
                    pause_event=self._pause_event,
                ),
                timeout=self.spec.timeout_secs,
            )
            if (
                self._current.cancel_token is not None
                and self._current.cancel_token.is_set()
            ):
                summary = run.final_text or "(cancelled)"
                outcome = TeammateStatus.CANCELLED
            else:
                summary = run.final_text or "(no output)"
                outcome = TeammateStatus.REPORTING
        except TimeoutError:
            summary = f"teammate timed out after {self.spec.timeout_secs:.0f}s"
            outcome = TeammateStatus.TIMEOUT
        except Exception as e:
            summary = f"teammate error: {type(e).__name__}: {e}"
            outcome = TeammateStatus.FAILED

        self._transition(outcome, reason=f"agent {outcome.value}")

        kind = {
            TeammateStatus.REPORTING: "work_done",
            TeammateStatus.TIMEOUT: "work_timeout",
            TeammateStatus.FAILED: "work_failed",
            TeammateStatus.CANCELLED: "work_cancelled",
        }[outcome]

        await self._bus.send(
            self.name, self._lead_name, summary,
            kind=kind,
            meta={"task_id": self._current.task_id} if self._current.task_id else {},
        )

        if self._current.task_id and self._task_store:
            with contextlib.suppress(Exception):
                if outcome == TeammateStatus.REPORTING:
                    self._task_store.complete(
                        self._current.task_id, owner=self.name,
                    )
                elif outcome == TeammateStatus.TIMEOUT:
                    self._task_store.timeout(
                        self._current.task_id, owner=self.name,
                    )
                elif outcome == TeammateStatus.CANCELLED:
                    self._task_store.cancel(
                        self._current.task_id,
                        owner=self.name,
                        reason=summary,
                    )
                else:
                    self._task_store.fail(
                        self._current.task_id,
                        owner=self.name,
                        error=summary,
                    )

        self._cleanup_worktree()

        self._transition(
            TeammateStatus.IDLE, reason=f"recover from {outcome.value}"
        )
        self._current.clear()

    async def _submit_and_wait_plan(self) -> bool:
        plan_req = await self._plan_store.create(
            sender=self.name,
            plan=self._current.assignment or "",
            task_id=self._current.task_id,
            work_version=self._current.work_version,
        )
        await self._bus.send(
            self.name, self._lead_name,
            f"Plan submitted for review:\n{self._current.assignment}",
            kind="plan_submitted",
            meta={"request_id": plan_req.request_id},
        )
        self._transition(
            TeammateStatus.WAITING_APPROVAL, reason="plan submitted"
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
        if not self._worktree_enabled or not self._base_dir or not self._current.task_id:
            return None
        from .worktree import create_worktree

        wt_name = f"{self.name}_{self._current.work_version}"
        wt = create_worktree(wt_name, self._current.task_id, self._base_dir)
        self._worktree = wt
        return wt

    def _cleanup_worktree(self) -> None:
        if not self._worktree or not self._base_dir:
            return
        from .worktree import CleanupPolicy, WorktreeError, remove_worktree

        wt_name = f"{self.name}_{self._current.work_version}"
        with contextlib.suppress(WorktreeError):
            remove_worktree(wt_name, self._base_dir, policy=CleanupPolicy.SAFE)
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
                self._current.assignment = t.description or t.subject
                self._current.task_id = t.id
                self._current.work_version += 1
                self._current.started_at = time.monotonic()
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
                self._force_stop = True
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    def info(self) -> TeammateInfo:
        last = None
        if self._last_transition is not None:
            last = (
                f"{self._last_transition.from_status.value}"
                f"->{self._last_transition.to_status.value}"
                f" ({self._last_transition.reason})"
            )
        return TeammateInfo(
            name=self.name,
            role=self.spec.role,
            status=self._status.value,
            task_id=self._current.task_id,
            worktree=str(self._worktree) if self._worktree else None,
            last_transition=last,
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

    async def send_instruction(self, to: str, content: str) -> str:
        """lead 向运行中 teammate 发送运行时指令。"""
        if to not in self._teammates or not self._teammates[to].alive:
            raise ValueError(f"no active teammate: {to}")
        await self._bus.send(LEAD_NAME, to, content, kind="instruction")
        self._teammates[to].notify()
        return f"Instruction sent to {to}"

    async def assign_task(self, to: str, content: str) -> str:
        """lead 向 teammate 分配新任务（排队，不覆盖当前任务）。"""
        if to not in self._teammates or not self._teammates[to].alive:
            raise ValueError(f"no active teammate: {to}")
        await self._bus.send(LEAD_NAME, to, content, kind="assignment")
        self._teammates[to].notify()
        return f"Task assigned to {to}"

    async def cancel_teammate(self, name: str) -> str:
        """取消 teammate 当前任务。"""
        rt = self._teammates.get(name)
        if rt is None or not rt.alive:
            raise ValueError(f"no active teammate: {name}")
        await self._bus.send(LEAD_NAME, name, "cancel", kind="cancel")
        rt.notify()
        return f"Cancel signal sent to {name}"

    async def pause_teammate(self, name: str) -> str:
        """暂停 teammate 执行。"""
        rt = self._teammates.get(name)
        if rt is None or not rt.alive:
            raise ValueError(f"no active teammate: {name}")
        await self._bus.send(LEAD_NAME, name, "pause", kind="pause")
        rt.notify()
        return f"Pause signal sent to {name}"

    async def resume_teammate(self, name: str) -> str:
        """恢复 teammate 执行。"""
        rt = self._teammates.get(name)
        if rt is None or not rt.alive:
            raise ValueError(f"no active teammate: {name}")
        await self._bus.send(LEAD_NAME, name, "resume", kind="resume")
        rt.notify()
        return f"Resume signal sent to {name}"

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


class SendInstructionTool:
    """send_instruction：lead 向运行中 teammate 发送运行时指令。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "send_instruction"

    @property
    def description(self) -> str:
        return (
            "Send a runtime instruction to an active teammate. "
            "The instruction is injected between iterations without "
            "interrupting the current task."
        )

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
            result = await self._manager.send_instruction(to, content)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class AssignTaskTool:
    """assign_task：lead 向 teammate 分配新任务（排队）。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "assign_task"

    @property
    def description(self) -> str:
        return (
            "Assign a new task to an active teammate. "
            "The task is queued and executed after the current task finishes."
        )

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
            result = await self._manager.assign_task(to, content)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class CancelTeammateTool:
    """cancel_teammate：取消 teammate 当前任务。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "cancel_teammate"

    @property
    def description(self) -> str:
        return "Cancel the current task of an active teammate."

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
            result = await self._manager.cancel_teammate(teammate)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class PauseTeammateTool:
    """pause_teammate：暂停 teammate 执行。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "pause_teammate"

    @property
    def description(self) -> str:
        return "Pause an active teammate's execution."

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
            result = await self._manager.pause_teammate(teammate)
        except ValueError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(result)


class ResumeTeammateTool:
    """resume_teammate：恢复 teammate 执行。"""

    def __init__(self, manager: TeammateManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "resume_teammate"

    @property
    def description(self) -> str:
        return "Resume a paused teammate's execution."

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
            result = await self._manager.resume_teammate(teammate)
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
    registry.register(SendInstructionTool(manager))
    registry.register(AssignTaskTool(manager))
    registry.register(CancelTeammateTool(manager))
    registry.register(PauseTeammateTool(manager))
    registry.register(ResumeTeammateTool(manager))
    registry.register(RequestShutdownTool(manager))
    registry.register(ReviewPlanTool(manager.plan_store, manager.bus))


__all__ = [
    "AssignTaskTool",
    "CancelTeammateTool",
    "CreateWorktreeTool",
    "CurrentExecution",
    "LEAD_NAME",
    "ListTeammatesTool",
    "MessageBus",
    "PauseTeammateTool",
    "PlanStore",
    "RequestShutdownTool",
    "ResumeTeammateTool",
    "ReviewPlanTool",
    "SendInstructionTool",
    "SendToLeadTool",
    "SendTeammateMessageTool",
    "SpawnTeammateTool",
    "SubmitPlanTool",
    "TeammateInfo",
    "TeammateManager",
    "TeammateRuntime",
    "TeammateSpec",
    "TeammateStateError",
    "TeammateStatus",
    "TransitionRecord",
    "format_team_events",
    "register_team_tools",
]
