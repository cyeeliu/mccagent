"""事件源装配：按依赖序构造 12 个子系统并返回 EventSource。

从 __init__ 拆出的 wiring 层：每个 ``_wire_*`` 函数构造一个
子系统（enabled=False 时返回 None 或跳过副作用注册），
``build_source`` 按依赖顺序调用——字段即依赖图。行为与历史
_build_source 逐行等价；__init__ 保留 re-export 兼容。
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import click

from mcc_cli.events import AgentEventSource


def _system_prompt() -> str:
    from mcc_harness import SECURITY_SYSTEM_CLAUSE

    return (
        "You are mini-claude-code, an enterprise-grade AI coding assistant. "
        "Examine files with read_file or list_dir before making changes. "
        "Use edit_file for targeted modifications. "
        "Use run_bash for shell tasks. "
        "Use grep to search code. "
        "Be concise and precise.\n\n"
        + SECURITY_SYSTEM_CLAUSE
    )


def _build_hooks(config, work_dir: Path):
    """从配置构建钩子注册表：pre/post_tool_use 条目 → ShellHook。

    command 为脚本路径，相对路径基于工作目录解析。
    无钩子配置时返回 None。
    """
    from mcc_harness import HookRegistry, HookType

    entries = [
        (HookType.PRE_TOOL, e) for e in config.hooks.pre_tool_use
    ] + [
        (HookType.POST_TOOL, e) for e in config.hooks.post_tool_use
    ]
    if not entries:
        return None

    hook_registry = HookRegistry()
    for hook_type, entry in entries:
        script = Path(entry.command)
        if not script.is_absolute():
            script = work_dir / script
        hook_registry.add_shell(hook_type, script, matcher=entry.matcher)
    return hook_registry


def _parse_mock_script(path: str) -> list:
    """从 JSON 文件加载 MockLlm 脚本。"""
    from mcc_core import Usage
    from mcc_llm import MessageStop, TextDelta, ToolUseInputDelta, ToolUseStart

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = []
    for turn in data:
        events = []
        for ev in turn:
            kind = ev["kind"]
            if kind == "text":
                events.append(TextDelta(text=ev.get("text", "")))
            elif kind == "tool_use_start":
                events.append(ToolUseStart(id=ev["id"], name=ev["name"]))
            elif kind == "tool_use_input":
                events.append(ToolUseInputDelta(partial_json=json.dumps(ev["input"])))
            elif kind == "message_stop":
                events.append(
                    MessageStop(
                        stop_reason=ev.get("stop_reason", "end_turn"),
                        usage=Usage(**ev.get("usage", {})),
                    )
                )
        turns.append(events)
    return turns


# -- 各子系统装配 --


def _wire_config(work_dir: Path):
    """加载合并配置（home + project）。"""
    from mcc_config import ensure_default_config

    return ensure_default_config(work_dir)


def _wire_llm(config, mock: bool, mock_script: str | None, llm_base_url: str | None):
    """构建 LLM provider（mock / 真实 / 网关），按配置包限流。"""
    if mock or mock_script:
        from mcc_llm.mock import MockLlm

        if mock_script:
            turns = _parse_mock_script(mock_script)
        else:
            from mcc_llm.mock import single_text_turn

            turns = [single_text_turn("[mock] No --mock-script provided; echoing.")]
        llm = MockLlm(turns)
    else:
        # 网关与凭证约定统一在 mcc-config：显式 --llm-base-url >
        # settings.json llm 段 > MCC_LLM_BASE_URL，均未配置时按
        # 环境自动探测
        from mcc_config import build_llm_provider

        try:
            llm = build_llm_provider(base_url=llm_base_url, config=config)
        except RuntimeError as e:
            click.echo(f"Error: {e}\nUse --mock for testing without API keys.", err=True)
            sys.exit(2)

    if config.reliability.rpm > 0 or config.reliability.max_concurrency > 0:
        from mcc_harness import ThrottledLlm

        llm = ThrottledLlm(
            llm,
            rpm=config.reliability.rpm,
            max_concurrency=config.reliability.max_concurrency,
        )
    return llm


def _wire_sandbox_and_background(config, registry) -> tuple[Any, Any]:
    """沙箱执行器 + 后台任务管理器 + run_bash 工具注册。

    沙箱按配置生效（默认 none 即直跑，行为不变）；显式 bubblewrap
    不可用时在此报错退出，而不是静默降级。
    """
    from mcc_tools import RunBashTool
    from mcc_tools.sandbox import build_executor

    executor = build_executor(config.sandbox)

    background_manager = None
    if config.background.enabled:
        from mcc_harness import BackgroundTaskManager

        background_manager = BackgroundTaskManager(
            executor=executor,
            max_concurrent=config.background.max_concurrent,
            default_timeout=config.background.default_timeout,
        )

    registry.register(RunBashTool(executor, background_manager))

    if background_manager is not None:
        from mcc_harness import (
            BackgroundStatusTool,
            CancelBackgroundTaskTool,
        )

        registry.register(BackgroundStatusTool(background_manager))
        registry.register(CancelBackgroundTaskTool(background_manager))

    return executor, background_manager


def _wire_cron(config, work_dir: Path, registry):
    """Cron 调度器：按 cron 表达式定时触发 prompt。"""
    if not config.cron.enabled:
        return None

    from mcc_harness import (
        CancelCronTool,
        CronScheduler,
        ListCronsTool,
        ScheduleCronTool,
    )

    durable_path = work_dir / ".cron_jobs.json" if config.cron.durable else None
    scheduler = CronScheduler(
        durable_path=durable_path,
        poll_interval=config.cron.poll_interval,
    )
    registry.register(ScheduleCronTool(scheduler))
    registry.register(ListCronsTool(scheduler))
    registry.register(CancelCronTool(scheduler))
    return scheduler


def _wire_mcp(config, registry) -> None:
    """MCP server 连接与工具发现（stdio transport）。"""
    if not config.mcp_servers:
        return

    from mcc_mcp import McpClientManager

    mcp_manager = McpClientManager()
    servers = {k: v.model_dump() for k, v in config.mcp_servers.items()}
    asyncio.run(mcp_manager.connect_all(servers))
    mcp_manager.register_all(registry)


def _wire_todo_tasks(config, work_dir: Path, registry) -> tuple[Any, Any]:
    """todo_write（会话内轻量清单）+ task 工具集（持久化依赖图）。"""
    todo_manager = None
    if config.todo.enabled:
        from mcc_tools import TodoManager, TodoWriteTool

        todo_manager = TodoManager()
        registry.register(TodoWriteTool(todo_manager))

    task_store = None
    if config.tasks.enabled:
        from mcc_tools import (
            ClaimTaskTool,
            CompleteTaskTool,
            CreateTaskTool,
            GetTaskTool,
            ListTasksTool,
            TaskStore,
            UpdateTaskTool,
        )

        task_store = TaskStore(work_dir / ".tasks")
        registry.register(CreateTaskTool(task_store))
        registry.register(UpdateTaskTool(task_store))
        registry.register(ListTasksTool(task_store))
        registry.register(GetTaskTool(task_store))
        registry.register(ClaimTaskTool(task_store))
        registry.register(CompleteTaskTool(task_store))

    return todo_manager, task_store


def _wire_prompt_system(config, work_dir: Path, registry, no_skills: bool):
    """system prompt 组装：基线 + 记忆索引 + skills 目录；
    返回 (system, slash_resolver, skills_catalog)。"""
    from mcc_session import MarkdownMemoryStore, memory_tools

    system = _system_prompt()
    store = MarkdownMemoryStore()
    for tool in memory_tools(store):
        registry.register(tool)
    index = store.index_text()
    if index:
        system = f"{system}\n\n# Memory Index\n{index}"

    slash_resolver = None
    skills_catalog = None
    if config.skills.enabled and not no_skills:
        from functools import partial

        from mcc_skills import (
            SkillRouter,
            load_commands_merged,
            load_skills_merged,
            resolve_slash,
        )
        from mcc_tools import LoadSkillTool

        skills = load_skills_merged(
            Path.home() / ".mcc" / "skills",
            work_dir / "skills",
        )
        if skills:
            catalog = SkillRouter(skills).catalog_for_prompt()
            if catalog:
                system = f"{system}\n\n{catalog}"
                skills_catalog = catalog
            registry.register(LoadSkillTool(skills))

        commands = load_commands_merged(
            Path.home() / ".mcc" / "commands",
            work_dir / "commands",
        )
        if commands:
            slash_resolver = partial(resolve_slash, commands)

    return system, slash_resolver, skills_catalog


def _wire_subagent_workflow(
    config, work_dir: Path, llm, registry, hooks, subagent_model: str, summarize_model: str
) -> tuple[Any, Any]:
    """SubagentRunner + Workflow 运行时（可信代码编排多步调用）。"""
    from mcc_harness import SpawnSubagentTool, SubagentRunner, default_presets

    runner = SubagentRunner(
        llm=llm,
        registry=registry,
        concurrency=4,
        hooks=hooks,
    )
    registry.register(
        SpawnSubagentTool(runner, default_presets(subagent_model, summarize_model))
    )

    workflow_tool = None
    if config.workflow.enabled:
        from mcc_harness import SubagentWorkflowRunner, WorkflowTool

        _register_builtin_workflows()
        workflow_tool = WorkflowTool(
            SubagentWorkflowRunner(runner, subagent_model),
            work_dir / ".workflows",
            concurrency=config.workflow.concurrency,
            agent_cap=config.workflow.agent_cap,
            default_budget=config.workflow.budget_tokens,
        )
        registry.register(workflow_tool)

    return runner, workflow_tool


def _wire_team(
    config, work_dir: Path, llm, registry, task_store, hooks, checker, subagent_model: str
):
    """多 Agent 团队：长驻 teammate 协作。"""
    if not config.team.enabled:
        return None

    from mcc_harness import (
        MessageBus,
        PlanStore,
        TeammateManager,
        register_team_tools,
    )

    bus = MessageBus(work_dir)
    plan_store = PlanStore()
    team_manager = TeammateManager(
        llm=llm,
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        base_dir=work_dir,
        worktree_enabled=config.team.worktree_enabled,
        max_teammates=config.team.max_teammates,
        default_model=subagent_model,
        hooks=hooks,
        permission=checker,
    )
    register_team_tools(registry, team_manager)
    return team_manager


_REVIEW_FINDINGS_SCHEMA = {
    "type": "object",
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "severity"],
                "properties": {
                    "title": {"type": "string"},
                    "severity": {"type": "string",
                                 "enum": ["high", "medium", "low"]},
                },
            },
        }
    },
}
_REVIEW_VERDICT_SCHEMA = {
    "type": "object",
    "required": ["isReal", "reason"],
    "properties": {"isReal": {"type": "boolean"}, "reason": {"type": "string"}},
}


async def _review_changes_workflow(ctx, args):
    """按审查维度流水线（audit → verify-each），仅保留被确认的发现。"""
    import json as _json

    ctx.phase("Review")
    changes = args.get("changes", "")
    if not isinstance(changes, str):
        from mcc_harness import WorkflowInputError

        raise WorkflowInputError("args.changes must be a string")
    review_input = changes.strip() or "No change context was supplied."
    dimensions = ["correctness", "security", "performance", "style"]

    async def audit(_value, dimension, _idx):
        out = await ctx.agent(
            f"Review this change context for {dimension} issues. "
            "Report only issues supported by the supplied text.\n\n"
            f"{review_input}",
            schema=_REVIEW_FINDINGS_SCHEMA,
            label=f"audit:{dimension}",
            phase="Review",
        )
        return {"dimension": dimension, "findings": out["findings"]}

    async def verify(audited, dimension, _idx):
        ctx.phase("Verify")
        verdicts = await ctx.parallel([
            (lambda f=f: ctx.agent(
                f"Adversarially verify this {dimension} finding against the "
                f"supplied change context.\n\nChange context:\n{review_input}"
                f"\n\nFinding:\n{_json.dumps(f, ensure_ascii=False)}",
                schema=_REVIEW_VERDICT_SCHEMA,
                label=f"verify:{dimension}:{f['title']}",
                phase="Verify",
            ))
            for f in audited["findings"]
        ])
        confirmed = [f for f, v in zip(audited["findings"], verdicts, strict=True)
                     if v and v.get("isReal")]
        return {"dimension": dimension, "confirmed": confirmed}

    results = await ctx.pipeline(dimensions, audit, verify)
    confirmed = [
        {"dimension": r["dimension"], **f}
        for r in results if r for f in r["confirmed"]
    ]
    confirmed.sort(
        key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f["severity"], 3)
    )
    ctx.log(f"confirmed {len(confirmed)} real finding(s)")
    return {"confirmed": confirmed}


def _register_builtin_workflows() -> None:
    """注册内置 workflow：review-changes 多维审查。"""
    from mcc_harness import WORKFLOWS, register_workflow

    if "review-changes" in WORKFLOWS:
        return
    register_workflow(
        {
            "name": "review-changes",
            "description": "Review changed code across dimensions, "
            "verify each finding adversarially",
            "phases": ["Review", "Verify"],
        },
        _review_changes_workflow,
    )


def _wire_observability(config, work_dir: Path, sid: str, builder):
    """可观测性：JSON 日志落 .mcc/logs/ + 进程内指标聚合。"""
    if not config.observability.enabled:
        return None

    import logging as _logging

    from mcc_observability import (
        JsonFormatter,
        Metrics,
        ObservabilitySink,
    )

    log_dir = work_dir / ".mcc" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = _logging.FileHandler(log_dir / "mcc.log.jsonl", encoding="utf-8")
    handler.setFormatter(
        JsonFormatter(text=config.observability.log_format == "text")
    )
    logger = _logging.getLogger("mcc")
    logger.setLevel(config.observability.log_level.upper())
    if not any(isinstance(h, _logging.FileHandler) for h in logger.handlers):
        logger.addHandler(handler)
    logger.propagate = False
    sink = ObservabilitySink(
        logger,
        Metrics(),
        session_id=sid,
        sanitize_enabled=config.observability.sanitize,
    )
    builder.observability(sink)
    return sink


def _assemble_builder(
    config,
    llm,
    registry,
    ctx,
    work_dir: Path,
    system: str,
    recorder,
    *,
    model: str | None,
    no_compact: bool,
    history: list | None,
    hooks,
    todo_manager,
    task_store,
    background_manager,
    cron_scheduler,
    subagent_model: str,
    summarize_model: str,
):
    """AgentLoopBuilder 组装：基础层（模型/权限/安全/可靠性/预算）
    与特性层（goal/todo/background/cron）依序注入。"""
    from mcc_config import resolve_model
    from mcc_harness import AgentLoopBuilder, PermissionChecker

    builder = AgentLoopBuilder(llm, registry, ctx).system(system).recorder(recorder)

    if config.context.auto_compact and not no_compact:
        # 四步压缩管线：转存 → 归档 → 旧结果替换 → 模型摘要（最后手段）。
        # 转存目录在会话工作目录内，随产物自包含
        from mcc_harness import ContextCompactor

        builder = builder.compactor(ContextCompactor(
            persist_dir=work_dir / ".task_outputs" / "tool-results",
            char_limit=config.context.char_limit,
        ))

    builder = builder.model(resolve_model("main", model, config=config))
    builder = builder.summarize_model(summarize_model)

    checker = PermissionChecker(
        config.permissions, hardened=config.security.hardened
    )
    builder = builder.permission(checker)

    # 安全纵深：L3 hardened 由 checker 承载；L2/L5 注入 AgentLoop
    from mcc_harness import SecurityPolicy

    builder = builder.security(SecurityPolicy(
        hardened=config.security.hardened,
        redact_output=config.security.redact_output,
        untrusted_wrap=config.security.untrusted_wrap,
    ))

    # 可靠性：重试退避始终生效；熔断/turn 超时按配置启用
    from mcc_harness import CircuitBreaker, RetryPolicy

    builder = builder.retry_policy(RetryPolicy(
        max_attempts=config.reliability.max_attempts,
        base_delay=config.reliability.base_delay,
        max_delay=config.reliability.max_delay,
        jitter=config.reliability.jitter,
    ))
    if config.reliability.circuit_failure_threshold > 0:
        builder = builder.circuit_breaker(CircuitBreaker(
            failure_threshold=config.reliability.circuit_failure_threshold,
            reset_timeout=config.reliability.circuit_reset_timeout,
        ))
    if config.reliability.turn_timeout_secs > 0:
        builder = builder.turn_timeout(config.reliability.turn_timeout_secs)

    builder = builder.max_cost_usd(config.budget.max_usd_per_session)
    builder = builder.max_iterations(config.budget.max_iterations)
    builder = builder.warn_threshold(config.budget.warn_threshold)
    builder = builder.max_cost_per_turn(config.budget.max_usd_per_turn)

    if history:
        builder = builder.history(history)

    if hooks is not None:
        builder = builder.hooks(hooks)

    goal_controller = None
    if config.goal.enabled:
        from mcc_harness import GoalController, PromptGoalEvaluator

        evaluator_model = config.goal.evaluator_model or summarize_model
        evaluator = PromptGoalEvaluator(llm=llm, model=evaluator_model)
        goal_controller = GoalController(
            evaluator=evaluator,
            block_cap=config.goal.block_cap,
            max_goal_length=config.goal.max_goal_length,
        )
        builder = builder.goal(goal_controller)
        builder = builder.evaluator_model(evaluator_model)

    if todo_manager is not None:
        builder = builder.todo(todo_manager)
        builder = builder.todo_reminder_interval(config.todo.reminder_interval)
    if config.tasks.enabled:
        builder = builder.tasks_enabled(True)
    if background_manager is not None:
        builder = builder.background(background_manager)
    if cron_scheduler is not None:
        builder = builder.cron(cron_scheduler)

    return builder, checker, goal_controller


def build_source(
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    history: list | None = None,
    session_id: str | None = None,
    no_compact: bool = False,
    no_skills: bool = False,
    llm_base_url: str | None = None,
) -> Any:
    """按依赖序装配事件源。

    同时接线会话录制（JSONL 落盘）与长期记忆
    （save_memory / read_memory 工具 + system prompt 索引注入）。
    llm_base_url 显式指定时走 OpenAI 兼容协议的自托管网关。
    """
    from mcc_config import resolve_model
    from mcc_core import ToolContext
    from mcc_session import SessionRecorder
    from mcc_tools import default_registry

    config = _wire_config(work_dir)
    sid = session_id or uuid.uuid4().hex[:8]
    ctx = ToolContext(cwd=work_dir, session_id=sid)
    registry = default_registry()

    _executor, background_manager = _wire_sandbox_and_background(config, registry)
    cron_scheduler = _wire_cron(config, work_dir, registry)
    _wire_mcp(config, registry)
    todo_manager, task_store = _wire_todo_tasks(config, work_dir, registry)
    system, slash_resolver, skills_catalog = _wire_prompt_system(
        config, work_dir, registry, no_skills
    )
    llm = _wire_llm(config, mock, mock_script, llm_base_url)
    hook_registry = _build_hooks(config, work_dir)

    subagent_model = resolve_model("subagent", config=config)
    summarize_model = resolve_model("summarize", config=config)
    _runner, workflow_tool = _wire_subagent_workflow(
        config, work_dir, llm, registry, hook_registry,
        subagent_model, summarize_model,
    )

    recorder = SessionRecorder.open_sync(sid, work_dir)
    builder, checker, goal_controller = _assemble_builder(
        config, llm, registry, ctx, work_dir, system, recorder,
        model=model, no_compact=no_compact, history=history,
        hooks=hook_registry, todo_manager=todo_manager, task_store=task_store,
        background_manager=background_manager, cron_scheduler=cron_scheduler,
        subagent_model=subagent_model, summarize_model=summarize_model,
    )

    team_manager = _wire_team(
        config, work_dir, llm, registry, task_store, hook_registry,
        checker, subagent_model,
    )

    obs_sink = _wire_observability(config, work_dir, sid, builder)

    agent = builder.build()
    from mcc_cli.slash import build_default_registry

    return AgentEventSource(
        agent,
        slash_resolver=slash_resolver,
        goal_controller=goal_controller,
        team_manager=team_manager,
        cron_scheduler=cron_scheduler,
        workflow_tool=workflow_tool,
        observability_sink=obs_sink,
        slash_registry=build_default_registry(),
        config=config,
        work_dir=work_dir,
        skills_catalog=skills_catalog,
        session_short_id=sid,
    )
