"""Workflow 运行时：可信代码编排多步 agent 调用，模型按名触发。

workflow 是代码内注册的可信脚本（非模型自由编排），模型通过
``Workflow`` 工具按名调用并传 args。执行原语由 ExecutionState
注入脚本：agent()（子代理调用，journal 语义缓存支持断点续跑）、
parallel()（屏障并发）、pipeline()（无屏障流水线）、workflow()
（单层嵌套，共享 journal/budget/计数）。护栏：token Budget、
并发/agent 计数上限、schema 校验失败重试一次。每次 run 持久化
快照/输出/journal 到 ``.workflows/`` 目录，支持 resume。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mcc_core import Notice, Tool, ToolContext, ToolOutput


class WorkflowInputError(Exception):
    """workflow 参数、meta 或校验错误。"""


# -- run id --

_RUN_ID_RE = re.compile(r"^wf_[a-z0-9-]{1,40}_[0-9a-f]{8}$")


def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "-", name.lower())[:40].strip("-") or "run"


def create_run_id(meta: dict[str, Any]) -> str:
    return f"wf_{_slug(str(meta['name']))}_{secrets.token_hex(4)}"


def reserve_run_id(meta: dict[str, Any], store: Path) -> str:
    for _ in range(100):
        run_id = create_run_id(meta)
        if not (store / f"{run_id}.json").exists():
            return run_id
    raise WorkflowInputError("could not allocate a run id")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise WorkflowInputError(f"invalid run id: {run_id!r}")
    return run_id


# -- run lock --

_RUN_LOCKS: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def workflow_run_lock(run_id: str):
    """同一 run_id 的 run/resume 串行执行。"""
    lock = _RUN_LOCKS.setdefault(run_id, asyncio.Lock())
    async with lock:
        yield


# -- meta validation --


def validate_meta(meta: dict[str, Any]) -> None:
    if not isinstance(meta, dict):
        raise WorkflowInputError("workflow meta must be an object")
    name = meta.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise WorkflowInputError("workflow meta.name must be a slug string")
    if not isinstance(meta.get("description"), str) or not meta["description"]:
        raise WorkflowInputError("workflow meta.description must be a string")
    phases = meta.get("phases", [])
    if not isinstance(phases, list) or not all(isinstance(p, str) for p in phases):
        raise WorkflowInputError("workflow meta.phases must be a list of strings")


# -- JSON Schema subset --


class SimpleJsonSchema:
    """轻量 JSON Schema 子集校验：type/required/properties/items/enum。"""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def validate(
        self, value: Any, schema: dict[str, Any] | None = None
    ) -> tuple[bool, str | None]:
        schema = self.schema if schema is None else schema
        if not isinstance(schema, dict):
            return True, None
        expected = schema.get("type")
        if expected is not None and not self._check_type(value, expected):
            return False, f"expected type {expected}, got {type(value).__name__}"
        enum = schema.get("enum")
        if enum is not None and value not in enum:
            return False, f"value {value!r} not in enum {enum}"
        if expected == "object":
            for key in schema.get("required", []):
                if key not in value:
                    return False, f"missing required property: {key}"
            props = schema.get("properties", {})
            for key, subschema in props.items():
                if key in value:
                    ok, err = self.validate(value[key], subschema)
                    if not ok:
                        return False, f"property {key}: {err}"
        if expected == "array":
            item_schema = schema.get("items")
            if item_schema is not None:
                for idx, item in enumerate(value):
                    ok, err = self.validate(item, item_schema)
                    if not ok:
                        return False, f"item {idx}: {err}"
        return True, None

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        return {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float))
            and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "null": lambda v: v is None,
        }.get(expected, lambda v: True)(value)


def parse_json_value(text: str) -> Any:
    """从模型输出文本解析 JSON：先整体解析，失败则截取首个对象/数组。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [ln for ln in stripped.splitlines() if not ln.startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch in "{[":
            try:
                value, _end = decoder.raw_decode(stripped[idx:])
                return value
            except json.JSONDecodeError:
                continue
    raise WorkflowInputError("runner output is not valid JSON")


# -- Journal（断点续跑） --

MISS = object()


class WorkflowJournal:
    """append-only ``<run_id>.journal.jsonl``。

    agent() 的语义 key（kind|label|prompt|schema 哈希）命中缓存时
    resume 直接返回，不重跑子代理。
    """

    def __init__(self, run_id: str, store: Path, resume: bool = False) -> None:
        store.mkdir(parents=True, exist_ok=True)
        self.path = store / f"{run_id}.journal.jsonl"
        self.cache: dict[str, Any] = {}
        if resume:
            if not self.path.exists():
                raise WorkflowInputError(
                    f"resume journal not found for {run_id}"
                )
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if (
                        not isinstance(rec, dict)
                        or not isinstance(rec.get("key"), str)
                        or "value" not in rec
                    ):
                        raise ValueError("expected key/value record")
                except (json.JSONDecodeError, ValueError) as exc:
                    raise WorkflowInputError(
                        f"invalid resume journal record at line {line_number}"
                    ) from exc
                self.cache[rec["key"]] = rec["value"]
            self._f = self.path.open("a", encoding="utf-8")
        else:
            self._f = self.path.open("w", encoding="utf-8")

    def key(
        self, kind: str, label: str, prompt: str, schema: dict[str, Any] | None
    ) -> str:
        basis = f"{kind}|{label}|{prompt}|{json.dumps(schema, sort_keys=True)}"
        return f"{kind}-{_stable_int(basis) % 10**10:010d}"

    def cached(self, key: str) -> Any:
        return self.cache.get(key, MISS)

    def record(self, key: str, value: Any) -> None:
        self._f.write(json.dumps({"key": key, "value": value}, default=str) + "\n")
        self._f.flush()
        self.cache[key] = value

    def close(self) -> None:
        self._f.close()


# -- Budget / limits --


class Budget:
    """token 总量护栏：超额时 agent() 抛错而非静默超支。"""

    def __init__(self, total: int | None = None) -> None:
        self.total = total
        self._spent = 0

    def add(self, n: int) -> None:
        if self.total is not None and self._spent + n > self.total:
            raise WorkflowInputError(
                f"token budget exceeded ({self._spent + n} > {self.total})"
            )
        self._spent += n

    def spent(self) -> int:
        return self._spent

    def remaining(self) -> float:
        if self.total is None:
            return float("inf")
        return max(0, self.total - self._spent)


class ExecutionLimits:
    """run 级共享护栏（嵌套 workflow 共用）：并发信号量 + agent 计数。"""

    def __init__(self, concurrency: int = 4, agent_cap: int = 20) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)
        self.agent_cap = agent_cap
        self.agents = 0

    def claim_agent(self) -> None:
        self.agents += 1
        if self.agents > self.agent_cap:
            raise WorkflowInputError(f"agent() cap reached ({self.agent_cap})")


# -- Task lifecycle --


@dataclass
class WorkflowTask:
    """单次 workflow run 的状态、用量与进度事件。"""

    task_id: str
    run_id: str
    meta: dict[str, Any]
    status: str = "running"
    usage: dict[str, int] = field(
        default_factory=lambda: {"agents": 0, "tokens": 0}
    )
    progress: list[dict[str, Any]] = field(default_factory=list)

    def progress_event(self, ptype: str, **data: Any) -> None:
        self.progress.append({"type": ptype, **data})

    def serialize(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "taskType": "local_workflow",
            "runId": self.run_id,
            "workflowName": self.meta["name"],
            "status": self.status,
            "usage": dict(self.usage),
            "progress": list(self.progress),
        }


# -- Runner --


@dataclass
class RunnerOutput:
    value: Any
    tokens: int


class WorkflowRunner(Protocol):
    """agent() 原语的执行器协议。"""

    async def run(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        label: str,
        ctx: ToolContext,
    ) -> RunnerOutput: ...


class SubagentWorkflowRunner:
    """基于 SubagentRunner 的执行器：schema 模式注入 JSON 指令并校验重试。"""

    SYSTEM = (
        "You are a workflow subagent. Answer the task precisely and only "
        "with the requested output. Do not call tools unless required."
    )

    def __init__(
        self,
        runner: Any,
        model: str,
        *,
        max_iterations: int = 6,
        budget_tokens: int = 60_000,
        timeout_secs: float = 180.0,
    ) -> None:
        self._runner = runner
        self._model = model
        self._max_iterations = max_iterations
        self._budget_tokens = budget_tokens
        self._timeout_secs = timeout_secs

    async def run(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        label: str,
        ctx: ToolContext,
    ) -> RunnerOutput:
        from mcc_harness.subagent import SubagentSpec

        full_prompt = prompt
        if schema is not None:
            full_prompt = (
                prompt
                + "\n\nRespond with ONLY valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
            )
        spec = SubagentSpec(
            name=_slug(label),
            system=self.SYSTEM,
            model=self._model,
            tools=[],
            max_iterations=self._max_iterations,
            budget_tokens=self._budget_tokens,
            timeout_secs=self._timeout_secs,
        )
        result = await self._runner.run(spec, full_prompt, ctx)
        tokens = result.usage.input_tokens + result.usage.output_tokens
        if schema is None:
            return RunnerOutput(value=result.summary, tokens=tokens)

        ok, value, err = self._parse_and_validate(result.summary, schema)
        if not ok:
            retry_prompt = full_prompt + "\n\nReturn valid JSON."
            retry = await self._runner.run(spec, retry_prompt, ctx)
            tokens += retry.usage.input_tokens + retry.usage.output_tokens
            ok, value, err = self._parse_and_validate(retry.summary, schema)
            if not ok:
                raise WorkflowInputError(
                    f"agent({{schema}}) invalid output: {err}"
                )
        return RunnerOutput(value=value, tokens=tokens)

    @staticmethod
    def _parse_and_validate(
        text: str, schema: dict[str, Any]
    ) -> tuple[bool, Any, str | None]:
        try:
            value = parse_json_value(text)
        except WorkflowInputError as exc:
            return False, None, str(exc)
        ok, err = SimpleJsonSchema(schema).validate(value)
        return ok, value, err


# -- ExecutionState（注入脚本的编排原语） --


class ExecutionState:
    """workflow 脚本的编排上下文：agent/parallel/pipeline/workflow/phase/log。"""

    def __init__(
        self,
        task: WorkflowTask,
        journal: WorkflowJournal,
        runner: WorkflowRunner,
        budget: Budget,
        args: dict[str, Any],
        base_ctx: ToolContext,
        workflows: dict[str, tuple[dict[str, Any], Any]],
        event_sink: asyncio.Queue | None = None,
        depth: int = 0,
        limits: ExecutionLimits | None = None,
    ) -> None:
        self.task = task
        self.journal = journal
        self.runner = runner
        self.budget = budget
        self.args = args
        self.base_ctx = base_ctx
        self._workflows = workflows
        self._event_sink = event_sink
        self._depth = depth
        self._limits = limits or ExecutionLimits()
        self._phase: str | None = None
        self._phases_seen: set[str] = set()

    def phase(self, title: str) -> None:
        """进入阶段；同名阶段重复进入不重复发事件（pipeline 场景）。"""
        self._phase = title
        if title not in self._phases_seen:
            self._phases_seen.add(title)
            self._emit(f"[workflow] phase: {title}")
            self.task.progress_event("workflow_phase", title=title)

    def log(self, message: str) -> None:
        """记录一行 workflow 日志。"""
        self._emit(f"[workflow] {message}")
        self.task.progress_event("workflow_log", message=message)

    async def agent(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        label: str | None = None,
        phase: str | None = None,
    ) -> Any:
        """调用一个子代理；schema 模式校验输出（失败重试一次）。

        resume 时语义 key 命中 journal 缓存则直接返回，不重跑。
        """
        label = label or (prompt[:24] + "...")
        self._limits.claim_agent()
        if self.budget.remaining() <= 0:
            raise WorkflowInputError("token budget exceeded")

        key = self.journal.key("agent", label, prompt, schema)
        cached = self.journal.cached(key)
        if cached is not MISS:
            if schema is not None:
                ok, err = SimpleJsonSchema(schema).validate(cached)
                if not ok:
                    raise WorkflowInputError(
                        f"cached agent output failed schema validation: {err}"
                    )
            self.task.progress_event(
                "workflow_agent", label=label, phase=phase or self._phase,
                status="cached",
            )
            return cached

        async with self._limits.semaphore:
            run = await self.runner.run(prompt, schema, label, self.base_ctx)

        self.budget.add(run.tokens)
        self.task.usage["agents"] += 1
        self.task.usage["tokens"] += run.tokens
        self.journal.record(key, run.value)
        self._emit(f"[workflow] agent done: {label} ({run.tokens} tokens)")
        self.task.progress_event(
            "workflow_agent", label=label, phase=phase or self._phase,
            status="done",
        )
        return run.value

    async def parallel(self, thunks: list[Any]) -> list[Any]:
        """屏障并发：任一 thunk 失败则整体失败。"""
        return await asyncio.gather(*[thunk() for thunk in thunks])

    async def pipeline(self, items: list[Any], *stages: Any) -> list[Any]:
        """无屏障流水线：item 独立穿过多阶段，A 可在阶段 3 时 B 仍在阶段 1。"""

        async def run_item(item: Any, idx: int) -> Any:
            value = item
            for stage in stages:
                value = await stage(value, item, idx)
            return value

        return await asyncio.gather(
            *[run_item(item, idx) for idx, item in enumerate(items)]
        )

    async def workflow(self, name: str, args: dict[str, Any] | None = None) -> Any:
        """嵌套调用已注册的 workflow（仅一层），共享 journal/budget/limits。"""
        if self._depth >= 1:
            raise WorkflowInputError("workflow() nesting is one level only")
        if name not in self._workflows:
            raise WorkflowInputError(f"unknown workflow '{name}'")
        meta, fn = self._workflows[name]
        child = ExecutionState(
            self.task, self.journal, self.runner, self.budget,
            args or {}, self.base_ctx, self._workflows,
            event_sink=self._event_sink,
            depth=self._depth + 1, limits=self._limits,
        )
        return await fn(child, args or {})

    def _emit(self, text: str) -> None:
        if self._event_sink is not None:
            self._event_sink.put_nowait(Notice(text=text))


# -- workflow 注册表 --

WORKFLOWS: dict[str, tuple[dict[str, Any], Any]] = {}


def register_workflow(meta: dict[str, Any], fn: Any) -> None:
    """注册可信 workflow 脚本：meta 校验通过后按名可查。"""
    validate_meta(meta)
    WORKFLOWS[meta["name"]] = (meta, fn)


def list_workflows() -> list[dict[str, Any]]:
    return [meta for meta, _ in WORKFLOWS.values()]


# -- WorkflowTool --


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_snapshot(store: Path, run_id: str) -> dict[str, Any]:
    path = store / f"{run_id}.json"
    if not path.exists():
        raise WorkflowInputError(f"resume snapshot not found for {run_id}")
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowInputError(
            f"invalid resume snapshot for {run_id}"
        ) from exc
    if not isinstance(snapshot, dict):
        raise WorkflowInputError(f"invalid resume snapshot for {run_id}")
    return snapshot


class WorkflowTool(Tool):
    """Workflow 工具：模型按名调用已注册 workflow，支持 resume。

    v1 在 agent turn 内同步执行；长 workflow 可后续结合后台任务。
    """

    def __init__(
        self,
        runner: WorkflowRunner,
        store_dir: Path,
        *,
        workflows: dict[str, tuple[dict[str, Any], Any]] | None = None,
        event_sink: asyncio.Queue | None = None,
        concurrency: int = 4,
        agent_cap: int = 20,
        default_budget: int | None = None,
    ) -> None:
        self._runner = runner
        self._store = store_dir
        self._workflows = workflows if workflows is not None else WORKFLOWS
        self._event_sink = event_sink
        self._concurrency = concurrency
        self._agent_cap = agent_cap
        self._default_budget = default_budget

    def set_event_sink(self, sink: asyncio.Queue | None) -> None:
        """绑定/解绑事件队列（TUI 模式下进度实时可见）。"""
        self._event_sink = sink

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def description(self) -> str:
        return (
            "Run a registered workflow by name. Workflows are trusted "
            "multi-step orchestration scripts. Pass input via args; "
            "resume a failed run with resume_from_run_id."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Registered workflow name"},
                "args": {"type": "object", "description": "Workflow input arguments"},
                "resume_from_run_id": {
                    "type": "string",
                    "description": "Resume a previous run by run id",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def execute(
        self, input: dict[str, Any], ctx: ToolContext
    ) -> ToolOutput:
        try:
            out = await self.call(
                input.get("name"),
                input.get("args"),
                input.get("resume_from_run_id"),
                ctx,
            )
        except WorkflowInputError as exc:
            return ToolOutput.err(f"Error: {exc}")
        return ToolOutput.ok(json.dumps(out, ensure_ascii=False, default=str))

    async def call(
        self,
        name: Any,
        args: Any = None,
        resume_from_run_id: str | None = None,
        ctx: ToolContext | None = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str):
            raise WorkflowInputError("workflow name must be a string")
        if name not in self._workflows:
            raise WorkflowInputError(f"unknown workflow '{name}'")
        if args is not None and not isinstance(args, dict):
            raise WorkflowInputError("workflow args must be an object")
        if ctx is None:
            raise WorkflowInputError("tool context is required")
        meta, script_fn = self._workflows[name]

        resuming = resume_from_run_id is not None
        run_id = (
            validate_run_id(resume_from_run_id)
            if resuming
            else reserve_run_id(meta, self._store)
        )
        async with workflow_run_lock(run_id):
            return await self._call_locked(
                meta, script_fn, args, run_id, resuming, ctx
            )

    async def _call_locked(
        self,
        meta: dict[str, Any],
        script_fn: Any,
        args: dict[str, Any] | None,
        run_id: str,
        resuming: bool,
        ctx: ToolContext,
    ) -> dict[str, Any]:
        if resuming:
            snapshot = _read_snapshot(self._store, run_id)
            if snapshot.get("workflowName") != meta["name"]:
                raise WorkflowInputError(
                    "resume run id does not match workflow meta"
                )
            saved_args = snapshot.get("args", {})
            if args is None:
                args = saved_args
            elif args != saved_args:
                raise WorkflowInputError(
                    "resume args do not match the original run"
                )
        else:
            args = args or {}

        journal = WorkflowJournal(run_id, self._store, resume=resuming)
        task = WorkflowTask(
            task_id=f"{run_id}-task", run_id=run_id, meta=meta
        )
        task.progress_event(
            "task_started",
            workflow=meta["name"],
            resume=resuming,
        )
        _write_json(
            self._store / f"{run_id}.json",
            {"runId": run_id, "workflowName": meta["name"],
             "args": args, "task": task.serialize()},
        )

        budget_total = args.get("budget", self._default_budget)
        state = ExecutionState(
            task,
            journal,
            self._runner,
            Budget(budget_total),
            args,
            ctx,
            self._workflows,
            event_sink=self._event_sink,
            limits=ExecutionLimits(self._concurrency, self._agent_cap),
        )
        try:
            result = await script_fn(state, args)
            task.status = "completed"
        except Exception as exc:
            task.status = "failed"
            result = {"error": str(exc)}
        finally:
            journal.close()

        _write_json(self._store / f"{run_id}.output.json", result)
        _write_json(
            self._store / f"{run_id}.json",
            {"runId": run_id, "workflowName": meta["name"],
             "args": args, "task": task.serialize()},
        )
        (self._store / "last_run.txt").write_text(run_id, encoding="utf-8")
        self._emit(
            f"[workflow] {meta['name']} {task.status}: "
            f"{task.usage['agents']} agent(s), {task.usage['tokens']} tokens "
            f"(run {run_id})"
        )
        task.progress_event("task_notification", status=task.status)
        return {
            "launched": {
                "status": "async_launched",
                "taskId": task.task_id,
                "runId": run_id,
                "workflowName": meta["name"],
            },
            "result": result,
            "task": task.serialize(),
        }

    def _emit(self, text: str) -> None:
        if self._event_sink is not None:
            self._event_sink.put_nowait(Notice(text=text))
