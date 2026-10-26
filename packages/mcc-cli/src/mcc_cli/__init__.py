"""mcc 命令行入口。

解析参数并分发到 TUI 或 headless 模式。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from pathlib import Path

import click

from mcc_cli.events import AgentEventSource, EchoEventSource, EventSource
from mcc_cli.headless import run_headless


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("mcc-cli")
    except Exception:
        return "0.1.0"


def _system_prompt() -> str:
    return (
        "You are mini-claude-code, an enterprise-grade AI coding assistant. "
        "Examine files with read_file or list_dir before making changes. "
        "Use edit_file for targeted modifications. "
        "Use run_bash for shell tasks. "
        "Use grep to search code. "
        "Be concise and precise."
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


def _build_source(
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    history: list | None = None,
    session_id: str | None = None,
    no_compact: bool = False,
    no_skills: bool = False,
    llm_base_url: str | None = None,
) -> EventSource:
    """构建事件源：mock 模式用 MockLlm，否则用真实 LLM。

    同时接线会话录制（JSONL 落盘）与长期记忆
    （save_memory / read_memory 工具 + system prompt 索引注入）。
    llm_base_url 显式指定时走 OpenAI 兼容协议的自托管网关。
    """
    from mcc_config import ensure_default_config
    from mcc_core import ToolContext
    from mcc_harness import (
        AgentLoopBuilder,
        PermissionChecker,
        SpawnSubagentTool,
        SubagentRunner,
        default_presets,
    )
    from mcc_session import MarkdownMemoryStore, SessionRecorder, memory_tools
    from mcc_tools import RunBashTool, default_registry
    from mcc_tools.sandbox import build_executor

    config = ensure_default_config(work_dir)
    sid = session_id or uuid.uuid4().hex[:8]
    ctx = ToolContext(cwd=work_dir, session_id=sid)
    registry = default_registry()
    # 沙箱按配置生效（默认 none 即直跑，行为不变）；显式 bubblewrap
    # 不可用时在此报错退出，而不是静默降级
    sandbox_executor = build_executor(config.sandbox)

    # 后台任务管理器：异步执行长耗时命令
    background_manager = None
    if config.background.enabled:
        from mcc_harness import BackgroundTaskManager

        background_manager = BackgroundTaskManager(
            executor=sandbox_executor,
            max_concurrent=config.background.max_concurrent,
            default_timeout=config.background.default_timeout,
        )

    registry.register(RunBashTool(sandbox_executor, background_manager))

    if background_manager is not None:
        from mcc_harness import (
            BackgroundStatusTool,
            CancelBackgroundTaskTool,
        )

        registry.register(BackgroundStatusTool(background_manager))
        registry.register(CancelBackgroundTaskTool(background_manager))

    # Cron 调度器：按 cron 表达式定时触发 prompt
    cron_scheduler = None
    if config.cron.enabled:
        from mcc_harness import CronScheduler

        durable_path = work_dir / ".cron_jobs.json" if config.cron.durable else None
        cron_scheduler = CronScheduler(
            durable_path=durable_path,
            poll_interval=config.cron.poll_interval,
        )

    if cron_scheduler is not None:
        from mcc_harness import (
            CancelCronTool,
            ListCronsTool,
            ScheduleCronTool,
        )

        registry.register(ScheduleCronTool(cron_scheduler))
        registry.register(ListCronsTool(cron_scheduler))
        registry.register(CancelCronTool(cron_scheduler))

    # MCP server 连接与工具发现（stdio transport）
    if config.mcp_servers:
        from mcc_mcp import McpClientManager

        mcp_manager = McpClientManager()
        servers = {k: v.model_dump() for k, v in config.mcp_servers.items()}
        asyncio.run(mcp_manager.connect_all(servers))
        mcp_manager.register_all(registry)

    # todo_write：会话内轻量任务清单
    todo_manager = None
    if config.todo.enabled:
        from mcc_tools import TodoManager, TodoWriteTool

        todo_manager = TodoManager()
        registry.register(TodoWriteTool(todo_manager))

    # task 工具集：持久化任务记录与依赖图
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

    system = _system_prompt()
    store = MarkdownMemoryStore()
    for tool in memory_tools(store):
        registry.register(tool)
    index = store.index_text()
    if index:
        system = f"{system}\n\n# Memory Index\n{index}"

    slash_resolver = None
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
            registry.register(LoadSkillTool(skills))

        commands = load_commands_merged(
            Path.home() / ".mcc" / "commands",
            work_dir / "commands",
        )
        if commands:
            slash_resolver = partial(resolve_slash, commands)

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

    hook_registry = _build_hooks(config, work_dir)

    from mcc_config import resolve_model

    subagent_model = resolve_model("subagent", config=config)
    summarize_model = resolve_model("summarize", config=config)
    runner = SubagentRunner(
        llm=llm,
        registry=registry,
        concurrency=4,
        hooks=hook_registry,
    )
    registry.register(
        SpawnSubagentTool(runner, default_presets(subagent_model, summarize_model))
    )

    # Workflow 运行时：可信代码编排多步 agent 调用，模型按名触发
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

    recorder = SessionRecorder.open_sync(sid, work_dir)
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

    checker = PermissionChecker(config.permissions)
    builder = builder.permission(checker)

    # 多 Agent 团队：长驻 teammate 协作
    team_manager = None
    if config.team.enabled:
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
            default_model=resolve_model("subagent", config=config),
            hooks=hook_registry,
            permission=checker,
        )
        register_team_tools(registry, team_manager)

    builder = builder.max_cost_usd(config.budget.max_usd_per_session)
    builder = builder.max_iterations(config.budget.max_iterations)
    builder = builder.warn_threshold(config.budget.warn_threshold)
    builder = builder.max_cost_per_turn(config.budget.max_usd_per_turn)

    if history:
        builder = builder.history(history)

    if hook_registry is not None:
        builder = builder.hooks(hook_registry)

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

    agent = builder.build()
    return AgentEventSource(
        agent,
        slash_resolver=slash_resolver,
        goal_controller=goal_controller,
        team_manager=team_manager,
        cron_scheduler=cron_scheduler,
        workflow_tool=workflow_tool,
    )


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


def _history_events(history: list, cost_usd: float = 0.0) -> list:
    """把恢复的消息历史转换为 TUI 重放事件序列。

    历史只传给 AgentLoop 时 LLM 具备上下文，但渲染层从未
    收到事件，界面会是空白；此处按 user/assistant/tool 消息
    逐条还原轨迹，末尾以 TurnEnd 收尾（携带会话累计成本）。
    """
    from mcc_core import Role, TextBlock, ToolResultBlock, ToolUseBlock
    from mcc_core.event import TextDelta, ToolCallEnd, ToolCallStart, TurnEnd, UserEcho

    events = []
    for m in history:
        if m.role == Role.user:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(UserEcho(text=b.text))
        elif m.role == Role.assistant:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    events.append(TextDelta(text=b.text))
                elif isinstance(b, ToolUseBlock):
                    events.append(ToolCallStart(
                        id=b.id, name=b.name,
                        args_preview=json.dumps(b.input, ensure_ascii=False)
                        if b.input else "",
                    ))
        elif m.role == Role.tool:
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    events.append(
                        ToolCallEnd(
                            id=b.tool_use_id,
                            output=b.content[:500],
                            is_error=b.is_error,
                        )
                    )
    events.append(TurnEnd(cost_usd=cost_usd))
    return events


async def _run_tui_mode(
    source: EventSource, replay: list | None = None, session_id: str | None = None
) -> int:
    from mcc_tui import TuiHandles, run_tui

    event_queue: asyncio.Queue = asyncio.Queue()
    input_queue: asyncio.Queue = asyncio.Queue()
    permission_queue: asyncio.Queue = asyncio.Queue()
    session_short_id = session_id or uuid.uuid4().hex[:8]

    agent = getattr(source, "agent", None)
    if agent is not None:
        agent.permission_responses = permission_queue

    for ev in replay or []:
        event_queue.put_nowait(ev)

    async def agent_worker() -> None:
        while True:
            user_input = await input_queue.get()
            try:
                async for ev in source.run(user_input):
                    await event_queue.put(ev)
            except Exception as exc:
                from mcc_core import Error as ErrorEvent

                await event_queue.put(ErrorEvent(text=str(exc)))
            team_mgr = getattr(source, "team_manager", None)
            if team_mgr is not None:
                inbox = await team_mgr.consume_lead_inbox()
                if inbox:
                    from mcc_harness import format_team_events

                    input_queue.put_nowait(format_team_events(inbox))

    worker = asyncio.create_task(agent_worker())
    cron = getattr(source, "cron_scheduler", None)
    if cron is not None:
        await cron.start()
    wf_tool = getattr(source, "workflow_tool", None)
    if wf_tool is not None:
        wf_tool.set_event_sink(event_queue)
    try:
        handles = TuiHandles(
            events=event_queue,
            input_tx=input_queue,
            session_short_id=session_short_id,
            permission_tx=permission_queue,
        )
        await run_tui(handles)
    finally:
        worker.cancel()
        if cron is not None:
            await cron.stop()
    return 0


def _parse_days(value: str | None) -> int | None:
    """解析 Nd 形式的天数，非法返回 None。"""
    if value is None:
        return None
    m = re.fullmatch(r"(\d+)d", value)
    return int(m.group(1)) if m else None


def _run_sessions_cmd(args: tuple[str, ...], older_than: str | None) -> None:
    """sessions 子命令：list / delete / purge。"""
    from mcc_session import (
        SessionNotFoundError,
        delete_session,
        list_sessions,
        purge_sessions,
    )

    sub = args[0] if args else "list"

    if sub == "list":
        metas = list_sessions()
        if not metas:
            click.echo("No sessions found.")
            return
        click.echo(f"{'ID':<10} {'STARTED':<17} {'TURNS':>5} {'COST':>9}  TITLE")
        for m in metas:
            started = m.started_at.strftime("%Y-%m-%d %H:%M")
            cost = f"${m.cost_usd:.4f}"
            title = m.title or ""
            click.echo(f"{m.id:<10} {started:<17} {m.turns:>5} {cost:>9}  {title}".rstrip())
        return

    if sub == "delete":
        if len(args) < 2:
            click.echo("Usage: mcc sessions delete <session-id>", err=True)
            sys.exit(2)
        try:
            meta = delete_session(args[1])
        except SessionNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        click.echo(f"deleted session {meta.id} ({meta.turns} turn(s))")
        return

    if sub == "purge":
        days = _parse_days(older_than)
        if days is None:
            click.echo("Usage: mcc sessions purge --older-than <Nd>", err=True)
            sys.exit(2)
        removed = purge_sessions(days)
        click.echo(f"purged {removed} session(s)")
        return

    click.echo(f"Unknown sessions subcommand: {sub}", err=True)
    sys.exit(2)


def _print_onboarding() -> None:
    """无 API key 时的首次运行引导。"""
    click.echo(
        "欢迎使用 mini-claude-code！\n"
        "\n"
        "尚未配置 LLM API key，请先设置其一：\n"
        "  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  export OPENAI_API_KEY=sk-...          # OpenAI 兼容端点\n"
        "\n"
        "自托管网关（vLLM / SGLang / Ollama 等）无需外部 key：\n"
        "  mcc --llm-base-url http://gateway:8000/v1 -p \"hi\"\n"
        "\n"
        "也可写入项目根目录 .env 文件（参考 .env.example）。\n"
        "无 key 体验完整链路：mcc --mock\n"
        "查阅文档：https://github.com/cyeeliu/mccagent",
        err=True,
    )


_COMPLETION_SHELLS = ("bash", "zsh", "fish")


def _run_completion_cmd(shell: str) -> None:
    """completion 子命令：输出 shell 补全脚本。"""
    from click.shell_completion import get_completion_class

    if shell not in _COMPLETION_SHELLS:
        click.echo(
            f"Unsupported shell: {shell} ({'/'.join(_COMPLETION_SHELLS)})", err=True
        )
        sys.exit(2)
    comp_cls = get_completion_class(shell)
    assert comp_cls is not None
    comp = comp_cls(main, {}, "mcc", "_MCC_COMPLETE")
    script = getattr(comp, "get_completion_script", None)
    click.echo(script() if script else comp.source())


def _latest_pypi_version(package: str = "mcc-cli") -> str | None:
    """查询 PyPI 上的最新发布版本，网络失败返回 None。"""
    import httpx

    try:
        resp = httpx.get(
            f"https://pypi.org/pypi/{package}/json", timeout=5.0
        )
        resp.raise_for_status()
        return resp.json()["info"]["version"]
    except Exception:
        return None


def _parse_version(v: str) -> tuple[int, ...]:
    """把 'a.b.c' 解析为可比较的整数元组，每段只取前导数字。"""
    parts = []
    for seg in v.strip().lstrip("v").split("."):
        m = re.match(r"\d+", seg)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def _is_newer_version(latest: str, current: str) -> bool:
    return _parse_version(latest) > _parse_version(current)


def _run_update_cmd() -> None:
    """update 子命令：比对 PyPI 最新版本并提示升级方式。"""
    current = _version()
    click.echo(f"mcc {current}")
    latest = _latest_pypi_version()
    if latest is None:
        click.echo("Could not check for updates (network unreachable).", err=True)
        return
    if _is_newer_version(latest, current):
        click.echo(f"New version available: {latest}")
        click.echo("Upgrade with:\n  uv tool upgrade mcc-cli\n  # or: pipx upgrade mcc-cli")
    else:
        click.echo("You are on the latest version.")


def _run_resume_cmd(
    args: tuple[str, ...],
    work_dir: Path,
    model: str | None,
    mock: bool,
    mock_script: str | None,
    llm_base_url: str | None = None,
    no_compact: bool = False,
    no_skills: bool = False,
) -> None:
    """resume 子命令：加载历史会话并继续对话。"""
    from mcc_session import SessionNotFoundError, load_session

    if not args:
        click.echo("Usage: mcc resume <session-id>", err=True)
        sys.exit(2)

    try:
        meta, history = load_session(args[0])
    except SessionNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not (mock or mock_script or llm_base_url
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")):
        _print_onboarding()
        sys.exit(2)

    click.echo(f"Resuming session {meta.id} ({meta.turns} turn(s), "
               f"${meta.cost_usd:.4f})" + (f" — {meta.title}" if meta.title else ""),
               err=True)
    source = _build_source(
        work_dir,
        model,
        mock,
        mock_script,
        history=history,
        session_id=meta.id,
        llm_base_url=llm_base_url,
        no_compact=no_compact,
        no_skills=no_skills,
    )
    replay = _history_events(history, cost_usd=meta.cost_usd)
    sys.exit(
        asyncio.run(_run_tui_mode(source, replay=replay, session_id=meta.id))
    )


@click.command()
@click.option("-p", "--prompt", "prompt", default=None, help="单次 headless 模式")
@click.option("--cwd", "cwd", default=None, envvar="MCC_PROJECT", help="工作目录")
@click.option("--model", "model", default=None, envvar="MODEL", help="覆盖模型")
@click.option("--quiet", "quiet", is_flag=True, help="只输出最终答案")
@click.option("--mock", "mock", is_flag=True, help="使用 Mock LLM（无需 API key）")
@click.option(
    "--mock-script",
    "mock_script",
    default=None,
    help="MockLlm 脚本 JSON 文件路径",
)
@click.option(
    "--no-compact",
    "no_compact",
    is_flag=True,
    help="禁用上下文自动压缩（长会话可能因上下文超限失败）",
)
@click.option(
    "--no-skills",
    "no_skills",
    is_flag=True,
    help="禁用技能与 slash 命令加载",
)
@click.option(
    "--llm-base-url",
    "llm_base_url",
    default=None,
    envvar="MCC_LLM_BASE_URL",
    help="OpenAI 兼容自托管网关地址（vLLM / SGLang / Ollama 等）",
)
@click.option(
    "--older-than",
    "older_than",
    default=None,
    hidden=True,
    help="sessions purge：清理 N 天前的会话（如 30d）",
)
@click.argument(
    "cmd",
    required=False,
    type=click.Choice(
        ["tui", "config", "version", "resume", "sessions", "completion", "update"]
    ),
)
@click.argument("args", nargs=-1)
def main(
    cmd: str | None,
    prompt: str | None,
    cwd: str | None,
    model: str | None,
    quiet: bool,
    mock: bool,
    mock_script: str | None,
    llm_base_url: str | None,
    no_compact: bool,
    no_skills: bool,
    older_than: str | None,
    args: tuple[str, ...],
) -> None:
    """mini-claude-code — 企业级 AI 编码助手。"""
    work_dir = Path(cwd) if cwd else Path.cwd()

    if cmd == "version":
        click.echo(f"mcc {_version()}")
        return

    if cmd == "config":
        from mcc_config import load

        config = asyncio.run(load(work_dir))
        click.echo(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return

    if cmd == "sessions":
        _run_sessions_cmd(args, older_than)
        return

    if cmd == "completion":
        if not args:
            click.echo("Usage: mcc completion <bash|zsh|fish>", err=True)
            sys.exit(2)
        _run_completion_cmd(args[0])
        return

    if cmd == "update":
        _run_update_cmd()
        return

    if cmd == "resume":
        _run_resume_cmd(args, work_dir, model, mock, mock_script, llm_base_url,
                         no_compact, no_skills)
        return

    has_llm = bool(
        mock or mock_script or llm_base_url
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if has_llm:
        source = _build_source(work_dir, model, mock, mock_script,
                               llm_base_url=llm_base_url,
                               no_compact=no_compact, no_skills=no_skills)
    elif prompt is not None:
        _print_onboarding()
        sys.exit(2)
    else:
        _print_onboarding()
        source = EchoEventSource()

    try:
        if cmd == "tui" or (cmd is None and prompt is None):
            sys.exit(asyncio.run(_run_tui_mode(source)))
        elif prompt is not None:
            sys.exit(asyncio.run(run_headless(source, prompt, quiet)))
    except KeyboardInterrupt:
        sys.exit(0)


__all__ = ["main"]
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("mcc-cli")
except Exception:
    __version__ = "0.1.0"
