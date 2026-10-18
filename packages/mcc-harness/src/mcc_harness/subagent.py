"""Subagent 并行执行：隔离上下文的子代理派生与预设库。

子代理复用 AgentLoop，在独立上下文中执行一次性任务，
只把精炼结论返还给主代理；工具注册表经过 subset 筛选，
子代理天然拿不到 spawn_subagent，配合深度上限防止递归派生。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mcc_core import ToolContext, ToolOutput, Usage
from mcc_llm import LlmProvider
from mcc_tools import ToolRegistry

from .agent import AgentLoop, model_price_per_1m
from .hooks import HookContext, HookRegistry, HookType
from .permission import PermissionChecker


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


@dataclass
class SubagentResult:
    """子代理运行结果：精炼结论与用量。"""

    name: str
    summary: str
    usage: Usage = field(default_factory=Usage)
    stopped_reason: str = "end_turn"


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
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self.hooks = hooks
        self.permission = permission

    async def run(
        self, spec: SubagentSpec, task: str, parent: ToolContext
    ) -> SubagentResult:
        """执行一个子代理：隔离上下文 + 超时 + 预算护栏。"""
        async with self._semaphore:
            ctx = ToolContext(
                cwd=parent.cwd,
                session_id=f"{parent.session_id}::{spec.name}",
                depth=parent.depth + 1,
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
            )

            try:
                run = await asyncio.wait_for(
                    agent.run(task), timeout=spec.timeout_secs
                )
            except TimeoutError:
                return SubagentResult(
                    name=spec.name,
                    summary=(
                        f"subagent timed out after {spec.timeout_secs:.0f}s; "
                        "partial work was cancelled"
                    ),
                    usage=Usage(),
                    stopped_reason="timeout",
                )

            if self.hooks:
                await self.hooks.run_hooks(
                    HookType.ON_SUBAGENT_STOP,
                    HookContext(
                        hook_type=HookType.ON_SUBAGENT_STOP,
                        session_id=ctx.session_id,
                        depth=ctx.depth,
                        cwd=ctx.cwd,
                    ),
                )

            return SubagentResult(
                name=spec.name,
                summary=run.final_text,
                usage=run.total_usage,
                stopped_reason="end_turn",
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

        content = (
            f"[subagent={r.name} tokens={r.usage.input_tokens}in/"
            f"{r.usage.output_tokens}out reason={r.stopped_reason}]\n{r.summary}"
        )
        if r.stopped_reason == "end_turn":
            return ToolOutput.ok(content)
        return ToolOutput.err(content)


__all__ = [
    "CODE_EXPLORER_SYSTEM",
    "SUMMARIZER_SYSTEM",
    "TEST_RUNNER_SYSTEM",
    "SpawnSubagentTool",
    "SubagentResult",
    "SubagentRunner",
    "SubagentSpec",
    "default_presets",
    "token_budget_to_usd",
]
