"""Subagent 并行执行：隔离上下文的子代理派生与预设库。

子代理复用 AgentLoop，在独立上下文中执行一次性任务，
只把精炼结论返还给主代理；工具注册表经过 subset 筛选，
子代理天然拿不到 spawn_subagent，配合深度上限防止递归派生。
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mcc_core import ToolContext, ToolOutput, Usage
from mcc_core.event import PermissionDecision
from mcc_core.event import PermissionRequest as PermissionRequestEvent
from mcc_llm import LlmProvider
from mcc_tools import ToolRegistry

from .agent import AgentLoop, model_price_per_1m, permission_request_for
from .hooks import HookContext, HookRegistry, HookType
from .permission import PermissionChecker


@dataclass
class PermissionChannelRequest:
    """通道中的权限请求，供外部消费后响应。"""

    request_id: int
    agent_id: str
    tool_name: str
    tool_input: dict[str, Any]
    category: str
    message: str


@dataclass
class PermissionAuditEntry:
    """权限审批审计记录。"""

    agent_id: str
    tool_name: str
    category: str
    decision: str
    reason: str = ""
    timestamp: float = 0.0


class PermissionChannel:
    """权限审批通道：子代理 ↔ 外部决策者。

    子代理运行时遇到 Ask 权限决策，通过通道发出请求。
    外部（父代理 / TUI）通过 respond() 响应。

    委托权限范围（auto_allow）内的请求自动允许，不阻塞等待；
    范围外的高风险操作转发到 pending 队列等外部响应。
    超时默认拒绝。所有决策记入审计日志。
    每个通道绑定唯一 agent_id，多个并发子代理互不串线。
    """

    def __init__(
        self,
        agent_id: str,
        *,
        auto_allow: frozenset[str] = frozenset(),
        timeout_secs: float = 30.0,
    ) -> None:
        self.agent_id = agent_id
        self._auto_allow = auto_allow
        self._timeout_secs = timeout_secs
        self._responses: asyncio.Queue[PermissionDecision] = asyncio.Queue()
        self._pending: asyncio.Queue[PermissionChannelRequest] = asyncio.Queue()
        self._audit: list[PermissionAuditEntry] = []

    @property
    def response_queue(self) -> asyncio.Queue[PermissionDecision]:
        """传给 AgentLoop 的 permission_responses 队列。"""
        return self._responses

    @property
    def pending(self) -> asyncio.Queue[PermissionChannelRequest]:
        """外部消费的待处理请求队列。"""
        return self._pending

    @property
    def timeout_secs(self) -> float:
        return self._timeout_secs

    async def handle_event(self, ev: PermissionRequestEvent) -> None:
        """处理从子代理 tx 队列提取的权限请求事件。

        委托范围内自动允许并记审计；否则转发到 pending 等外部响应。
        """
        preq = permission_request_for(ev.tool_name, ev.tool_input)
        if preq.category in self._auto_allow:
            self._audit.append(
                PermissionAuditEntry(
                    agent_id=self.agent_id,
                    tool_name=ev.tool_name,
                    category=preq.category,
                    decision="auto_allow",
                    timestamp=time.time(),
                )
            )
            await self._responses.put(
                PermissionDecision(request_id=ev.id, action="allow")
            )
        else:
            await self._pending.put(
                PermissionChannelRequest(
                    request_id=ev.id,
                    agent_id=self.agent_id,
                    tool_name=ev.tool_name,
                    tool_input=ev.tool_input,
                    category=preq.category,
                    message=ev.message,
                )
            )

    async def respond(self, decision: PermissionDecision) -> None:
        """外部响应权限请求，记入审计。"""
        self._audit.append(
            PermissionAuditEntry(
                agent_id=self.agent_id,
                tool_name="",
                category="",
                decision=decision.action,
                timestamp=time.time(),
            )
        )
        await self._responses.put(decision)

    def audit_log(self) -> list[PermissionAuditEntry]:
        """返回审计记录副本。"""
        return list(self._audit)


@dataclass
class SubagentSpec:
    """子代理规格：人格、模型、可用工具与三重护栏（迭代/预算/超时）。"""

    name: str
    system: str
    model: str
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    max_tokens: int = 2048
    budget_tokens: int = 80_000
    timeout_secs: float = 120.0


class SubagentStatus(StrEnum):
    """子代理生命周期状态。

    所有终止路径汇聚到统一出口，区分成功、超时、取消、
    预算超限与模型/工具异常。
    """

    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


@dataclass
class SubagentResult:
    """子代理运行结果：统一终止出口的状态与用量。

    usage 为 None 表示用量未知（超时/取消/异常时无法获取），
    不伪造为零消耗；成功时从 AgentRun.total_usage 填充真实值。
    """

    name: str
    summary: str
    status: SubagentStatus = SubagentStatus.RUNNING
    usage: Usage | None = None
    cost_usd: float = 0.0
    stopped_reason: str = "end_turn"
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    partial_result: str | None = None
    permission_audit: list[PermissionAuditEntry] = field(default_factory=list)


def token_budget_to_usd(budget_tokens: int, model: str) -> float:
    """token 总预算折算为单次运行的美元上限。

    按 input/output 单价均摊估算，保守起见不区分流向。
    """
    in_m, out_m = model_price_per_1m(model)
    return budget_tokens / 1_000_000.0 * (in_m + out_m) / 2.0


CODE_EXPLORER_SYSTEM = """\
你是代码探索员。给定任务：
1. 用 list_dir / grep 定位最多 5 个相关文件
2. 用 read_file 仔细阅读（可分页）
3. 输出结构化摘要：
   - 相关文件与行号
   - 每个文件的职责
   - 关键的类型/函数
   - 潜在问题
最终回答不超过 400 字，不要贴原代码块（除非确有必要，最多 10 行）。
如果不确定，明确说明 "I am not confident about ..."。"""

TEST_RUNNER_SYSTEM = """\
你运行测试并汇总失败：测试名、失败断言、最可能原因。
用 run_bash 执行测试命令，用 read_file 查看失败上下文。
最终回答不超过 300 字，并给出修复建议。
如果不确定，明确说明 "I am not confident about ..."。"""

SUMMARIZER_SYSTEM = (
    "你是一个只回答摘要的子 Agent。不要使用工具，"
    "直接基于给定任务压缩输出。"
)


def default_presets(
    subagent_model: str = "claude-haiku-4-5",
    summarize_model: str | None = None,
) -> dict[str, SubagentSpec]:
    """预设库：高频角色的开箱即用规格。

    模型由调用方解析后注入（主配置 subagent / summarize 类型，
    未配置时一律回落 main），预设库本身不感知配置。
    """
    return {
        "code_explorer": SubagentSpec(
            name="code_explorer",
            system=CODE_EXPLORER_SYSTEM,
            model=subagent_model,
            tools=["read_file", "list_dir", "grep"],
            max_iterations=10,
            max_tokens=2048,
            budget_tokens=80_000,
            timeout_secs=120.0,
        ),
        "test_runner": SubagentSpec(
            name="test_runner",
            system=TEST_RUNNER_SYSTEM,
            model=subagent_model,
            tools=["run_bash", "read_file"],
            max_iterations=10,
            max_tokens=2048,
            budget_tokens=150_000,
            timeout_secs=300.0,
        ),
        "summarizer": SubagentSpec(
            name="summarizer",
            system=SUMMARIZER_SYSTEM,
            model=summarize_model or subagent_model,
            tools=[],
            max_iterations=1,
            max_tokens=1024,
            budget_tokens=30_000,
            timeout_secs=30.0,
        ),
    }


class SubagentRunner:
    """子代理执行器。

    信号量限制全局并发，超时与 token 预算换算成单次运行护栏；
    子代理共享主代理的 LLM、权限检查器与钩子注册表，
    但拥有独立消息上下文，原始对话不回传主代理。
    """

    def __init__(
        self,
        llm: LlmProvider,
        registry: ToolRegistry,
        concurrency: int = 4,
        hooks: HookRegistry | None = None,
        permission: PermissionChecker | None = None,
        permission_auto_allow: frozenset[str] = frozenset(),
        permission_timeout_secs: float = 30.0,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self.hooks = hooks
        self.permission = permission
        self._permission_auto_allow = permission_auto_allow
        self._permission_timeout_secs = permission_timeout_secs

    async def run(
        self,
        spec: SubagentSpec,
        task: str,
        parent: ToolContext,
        *,
        channel: PermissionChannel | None = None,
    ) -> SubagentResult:
        """执行一个子代理：隔离上下文 + 超时 + 预算护栏。

        所有终止路径（成功、超时、取消、预算超限、异常）汇聚到
        统一出口，保证 ON_SUBAGENT_STOP Hook 在每条路径触发且只触发一次。
        CancelledError 不在此捕获（BaseException），Hook 经 finally 触发
        后取消信号继续向上传播。

        权限审批通过 PermissionChannel 处理：委托范围内自动允许，
        范围外转发到通道 pending 队列等外部响应，超时默认拒绝。
        外部可传入预创建的 channel 以便在运行期间响应审批请求。
        """
        async with self._semaphore:
            ctx = ToolContext(
                cwd=parent.cwd,
                session_id=f"{parent.session_id}::{spec.name}",
                depth=parent.depth + 1,
            )

            if channel is None:
                channel = PermissionChannel(
                    agent_id=ctx.session_id,
                    auto_allow=self._permission_auto_allow,
                    timeout_secs=self._permission_timeout_secs,
                )

            agent = AgentLoop(
                llm=self.llm,
                registry=self.registry.subset(spec.tools),
                ctx=ctx,
                system=spec.system,
                model=spec.model,
                max_tokens=spec.max_tokens,
                max_iterations=spec.max_iterations,
                temperature=0.0,
                permission=self.permission,
                max_cost_usd=token_budget_to_usd(spec.budget_tokens, spec.model),
                hooks=self.hooks,
                permission_responses=channel.response_queue,
                permission_timeout_secs=self._permission_timeout_secs,
            )

            tx: asyncio.Queue = asyncio.Queue()

            async def pump_permission_events() -> None:
                """从子代理事件队列提取权限请求，通过通道处理。"""
                while True:
                    ev = await tx.get()
                    if isinstance(ev, PermissionRequestEvent):
                        await channel.handle_event(ev)

            pump_task = asyncio.create_task(pump_permission_events())

            started_at = time.monotonic()
            status = SubagentStatus.RUNNING
            summary = ""
            usage: Usage | None = None
            cost_usd = 0.0
            stopped_reason = "end_turn"
            error: str | None = None

            try:
                run = await asyncio.wait_for(
                    agent.run_streaming(task, tx), timeout=spec.timeout_secs
                )
                status = SubagentStatus.COMPLETED
                summary = run.final_text
                usage = run.total_usage
                cost_usd = run.cost_usd
            except TimeoutError:
                status = SubagentStatus.TIMEOUT
                summary = (
                    f"subagent timed out after {spec.timeout_secs:.0f}s; "
                    "partial work was cancelled"
                )
                stopped_reason = "timeout"
            except Exception as e:
                err_str = str(e)
                lowered = err_str.lower()
                if "budget" in lowered or "cost" in lowered or "max_cost" in lowered:
                    status = SubagentStatus.BUDGET_EXCEEDED
                    stopped_reason = "budget_exceeded"
                else:
                    status = SubagentStatus.ERROR
                    stopped_reason = "error"
                summary = f"subagent failed: {err_str}"
                error = err_str
            finally:
                pump_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump_task
                finished_at = time.monotonic()
                if self.hooks:
                    with contextlib.suppress(Exception):
                        await self.hooks.run_hooks(
                            HookType.ON_SUBAGENT_STOP,
                            HookContext(
                                hook_type=HookType.ON_SUBAGENT_STOP,
                                session_id=ctx.session_id,
                                depth=ctx.depth,
                                cwd=ctx.cwd,
                                cost_usd=cost_usd,
                                is_error=(status != SubagentStatus.COMPLETED),
                            ),
                        )

            return SubagentResult(
                name=spec.name,
                summary=summary,
                status=status,
                usage=usage,
                cost_usd=cost_usd,
                stopped_reason=stopped_reason,
                error=error,
                started_at=started_at,
                finished_at=finished_at,
                permission_audit=channel.audit_log(),
            )


class SpawnSubagentTool:
    """spawn_subagent 工具：按预设派生子代理。

    子代理的注册表来自 subset 筛选，不含本工具自身，
    深度上限作为第二道递归护栏。
    """

    def __init__(
        self,
        runner: SubagentRunner,
        presets: dict[str, SubagentSpec],
        max_depth: int = 3,
    ) -> None:
        self.runner = runner
        self.presets = presets
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "spawn_subagent"

    @property
    def description(self) -> str:
        return (
            "Delegate to a specialized subagent with its own isolated context. "
            "Available presets: code_explorer (read files+grep), "
            "test_runner (run tests), summarizer (pure LLM). "
            "Returns only a concise summary. "
            "Use when: reading many files for a small answer, "
            "OR parallelizing independent tasks."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["preset", "task"],
            "properties": {
                "preset": {"enum": sorted(self.presets.keys())},
                "task": {
                    "type": "string",
                    "description": "One-shot self-contained instruction",
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        if ctx.depth >= self.max_depth:
            return ToolOutput.err(
                f"subagent depth limit {self.max_depth} reached"
            )

        preset = str(input.get("preset", ""))
        task = str(input.get("task", ""))
        if not preset or not task:
            return ToolOutput.err("'preset' and 'task' are both required")

        spec = self.presets.get(preset)
        if spec is None:
            return ToolOutput.err(f"unknown preset: {preset}")

        try:
            r = await self.runner.run(spec, task, ctx)
        except Exception as e:
            return ToolOutput.err(f"subagent failed: {e}")

        if r.usage is not None:
            tokens = f"{r.usage.input_tokens}in/{r.usage.output_tokens}out"
        else:
            tokens = "unknown"

        content = (
            f"[subagent={r.name} status={r.status.value} "
            f"tokens={tokens} reason={r.stopped_reason}]\n{r.summary}"
        )
        if r.status == SubagentStatus.COMPLETED:
            return ToolOutput.ok(content)
        return ToolOutput.err(content)


__all__ = [
    "CODE_EXPLORER_SYSTEM",
    "SUMMARIZER_SYSTEM",
    "TEST_RUNNER_SYSTEM",
    "PermissionAuditEntry",
    "PermissionChannel",
    "PermissionChannelRequest",
    "SpawnSubagentTool",
    "SubagentResult",
    "SubagentRunner",
    "SubagentSpec",
    "SubagentStatus",
    "default_presets",
    "token_budget_to_usd",
]
