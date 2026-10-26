"""workflow 运行时测试：run_id / meta / Schema / Journal / Budget / 原语 / 工具。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcc_config.loader import merge
from mcc_config.models import Config, WorkflowConfig
from mcc_core import ToolContext, Usage
from mcc_harness import (
    MISS,
    Budget,
    ExecutionLimits,
    ExecutionState,
    RunnerOutput,
    SimpleJsonSchema,
    SubagentWorkflowRunner,
    WorkflowInputError,
    WorkflowJournal,
    WorkflowTask,
    WorkflowTool,
    create_run_id,
    parse_json_value,
    register_workflow,
    reserve_run_id,
    validate_meta,
    validate_run_id,
)
from mcc_harness.subagent import SubagentResult
from mcc_harness.workflow import _read_snapshot


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


class MockWorkflowRunner:
    """按 label 前缀决定输出的假执行器，记录调用次数。"""

    def __init__(self, handler: Any = None) -> None:
        self.calls: list[tuple[str, Any, str]] = []
        self._handler = handler

    async def run(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        label: str,
        ctx: ToolContext,
    ) -> RunnerOutput:
        self.calls.append((prompt, schema, label))
        if self._handler is not None:
            return self._handler(prompt, schema, label)
        return RunnerOutput(value={"label": label, "ok": True}, tokens=10)


META = {"name": "demo-wf", "description": "demo", "phases": ["A", "B"]}


async def _echo_workflow(state: ExecutionState, args: dict[str, Any]) -> Any:
    return await state.agent(f"echo {args.get('x', '')}", label="echo")


def _tool(tmp_path: Path, runner: MockWorkflowRunner, **kw: Any) -> WorkflowTool:
    workflows = {"demo-wf": (META, _echo_workflow)}
    return WorkflowTool(
        runner, tmp_path / ".workflows", workflows=workflows, **kw
    )


# -- run id --


class TestRunId:
    def test_format(self) -> None:
        run_id = create_run_id(META)
        assert validate_run_id(run_id) == run_id

    @pytest.mark.parametrize(
        "bad",
        ["", "wf", "wf_x", "wf_demo_zzzz", "wf_demo-wf_0123456g",
         "Wf_demo-wf_01234567", 123],
    )
    def test_validate_rejects(self, bad: Any) -> None:
        with pytest.raises(WorkflowInputError):
            validate_run_id(bad)

    def test_reserve_skips_existing(self, tmp_path: Path) -> None:
        store = tmp_path / "wf"
        store.mkdir()
        run_id = reserve_run_id(META, store)
        (store / f"{run_id}.json").write_text("{}", encoding="utf-8")
        assert reserve_run_id(META, store) != run_id


# -- meta --


class TestValidateMeta:
    def test_valid(self) -> None:
        validate_meta(META)

    @pytest.mark.parametrize(
        "meta",
        [
            {},
            {"name": "Demo", "description": "x"},
            {"name": "demo_wf", "description": "x"},
            {"name": "demo", "description": ""},
            {"name": "demo", "description": "x", "phases": "A"},
            {"name": "demo", "description": "x", "phases": [1]},
            "not-a-dict",
        ],
    )
    def test_invalid(self, meta: Any) -> None:
        with pytest.raises(WorkflowInputError):
            validate_meta(meta)


# -- SimpleJsonSchema --


class TestSimpleJsonSchema:
    @pytest.mark.parametrize(
        ("schema", "value"),
        [
            ({"type": "string"}, "hello"),
            ({"type": "number"}, 1.5),
            ({"type": "integer"}, 3),
            ({"type": "boolean"}, True),
            ({"type": "object"}, {"a": 1}),
            ({"type": "array"}, [1, 2]),
            ({"type": "null"}, None),
            ({"type": "string", "enum": ["a", "b"]}, "a"),
            ({"type": "object", "required": ["x"]}, {"x": 1}),
            ({"type": "object", "properties": {"x": {"type": "integer"}}},
             {"x": 1}),
            ({"type": "array", "items": {"type": "string"}}, ["a", "b"]),
            ({"type": "object",
              "properties": {"nested": {"type": "object",
                                        "required": ["y"]}}},
             {"nested": {"y": 2}}),
        ],
    )
    def test_valid(self, schema: dict, value: Any) -> None:
        ok, err = SimpleJsonSchema(schema).validate(value)
        assert ok
        assert err is None

    @pytest.mark.parametrize(
        ("schema", "value", "fragment"),
        [
            ({"type": "string"}, 1, "expected type string"),
            ({"type": "integer"}, True, "expected type integer"),
            ({"type": "object"}, [], "expected type object"),
            ({"type": "array"}, {}, "expected type array"),
            ({"type": "string", "enum": ["a"]}, "z", "not in enum"),
            ({"type": "object", "required": ["x"]}, {}, "missing required"),
            ({"type": "object", "properties": {"x": {"type": "integer"}}},
             {"x": "s"}, "property x"),
            ({"type": "array", "items": {"type": "string"}}, [1], "item 0"),
        ],
    )
    def test_invalid(self, schema: dict, value: Any, fragment: str) -> None:
        ok, err = SimpleJsonSchema(schema).validate(value)
        assert not ok
        assert fragment in err


class TestParseJsonValue:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"a": 1}', {"a": 1}),
            ("[1, 2]", [1, 2]),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('some text {"a": 1} trailing', {"a": 1}),
            ('noise [{"x": 1}] more', [{"x": 1}]),
        ],
    )
    def test_parse(self, text: str, expected: Any) -> None:
        assert parse_json_value(text) == expected

    def test_non_json_raises(self) -> None:
        with pytest.raises(WorkflowInputError):
            parse_json_value("no json here")


# -- Journal --


class TestWorkflowJournal:
    def test_key_stable_and_schema_order_independent(
        self, tmp_path: Path
    ) -> None:
        j = WorkflowJournal("wf_demo-wf_01234567", tmp_path)
        try:
            k1 = j.key("agent", "l", "p", {"a": 1, "b": 2})
            k2 = j.key("agent", "l", "p", {"b": 2, "a": 1})
            k3 = j.key("agent", "l", "p", None)
            assert k1 == k2
            assert k1 != k3
        finally:
            j.close()

    def test_record_and_cached(self, tmp_path: Path) -> None:
        j = WorkflowJournal("wf_demo-wf_01234567", tmp_path)
        j.record("agent-1", {"v": 1})
        assert j.cached("agent-1") == {"v": 1}
        assert j.cached("agent-2") is MISS
        j.close()
        lines = (tmp_path / "wf_demo-wf_01234567.journal.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        assert json.loads(lines[0]) == {"key": "agent-1", "value": {"v": 1}}

    def test_resume_loads_cache(self, tmp_path: Path) -> None:
        j = WorkflowJournal("wf_demo-wf_01234567", tmp_path)
        j.record("agent-1", 42)
        j.close()
        j2 = WorkflowJournal("wf_demo-wf_01234567", tmp_path, resume=True)
        try:
            assert j2.cached("agent-1") == 42
        finally:
            j2.close()

    def test_resume_missing_journal(self, tmp_path: Path) -> None:
        with pytest.raises(WorkflowInputError, match="journal not found"):
            WorkflowJournal("wf_demo-wf_01234567", tmp_path, resume=True)

    def test_resume_corrupt_line(self, tmp_path: Path) -> None:
        (tmp_path / "wf_demo-wf_01234567.journal.jsonl").write_text(
            '{"key": "k", "value": 1}\nnot json\n', encoding="utf-8"
        )
        with pytest.raises(WorkflowInputError, match="line 2"):
            WorkflowJournal("wf_demo-wf_01234567", tmp_path, resume=True)


# -- Budget / limits --


class TestBudget:
    def test_unlimited(self) -> None:
        b = Budget()
        b.add(10**9)
        assert b.remaining() == float("inf")

    def test_cap(self) -> None:
        b = Budget(total=100)
        b.add(60)
        assert b.spent() == 60
        assert b.remaining() == 40
        with pytest.raises(WorkflowInputError, match="budget exceeded"):
            b.add(41)


class TestExecutionLimits:
    def test_agent_cap(self) -> None:
        limits = ExecutionLimits(agent_cap=2)
        limits.claim_agent()
        limits.claim_agent()
        with pytest.raises(WorkflowInputError, match="cap reached"):
            limits.claim_agent()


# -- ExecutionState 原语 --


def _state(
    tmp_path: Path,
    runner: MockWorkflowRunner,
    *,
    budget: int | None = None,
    workflows: dict | None = None,
    agent_cap: int = 20,
) -> ExecutionState:
    task = WorkflowTask(task_id="t", run_id="wf_demo-wf_01234567", meta=META)
    journal = WorkflowJournal("wf_demo-wf_01234567", tmp_path)
    return ExecutionState(
        task,
        journal,
        runner,
        Budget(budget),
        {},
        _ctx(tmp_path),
        workflows if workflows is not None else {"demo-wf": (META, _echo_workflow)},
        limits=ExecutionLimits(agent_cap=agent_cap),
    )


class TestExecutionState:
    async def test_agent_records_journal_and_usage(
        self, tmp_path: Path
    ) -> None:
        runner = MockWorkflowRunner()
        state = _state(tmp_path, runner)
        value = await state.agent("do x", label="x")
        assert value == {"label": "x", "ok": True}
        assert len(runner.calls) == 1
        assert state.task.usage == {"agents": 1, "tokens": 10}
        assert len(state.journal.cache) == 1

    async def test_agent_cache_hit_skips_runner(self, tmp_path: Path) -> None:
        runner = MockWorkflowRunner()
        state = _state(tmp_path, runner)
        await state.agent("do x", label="x")
        value = await state.agent("do x", label="x")
        assert value == {"label": "x", "ok": True}
        assert len(runner.calls) == 1
        cached_events = [
            e for e in state.task.progress
            if e.get("status") == "cached"
        ]
        assert len(cached_events) == 1

    async def test_agent_budget_exceeded(self, tmp_path: Path) -> None:
        runner = MockWorkflowRunner()
        state = _state(tmp_path, runner, budget=5)
        with pytest.raises(WorkflowInputError, match="budget exceeded"):
            await state.agent("do x", label="x")
        assert len(runner.calls) == 1
        assert state.task.usage == {"agents": 0, "tokens": 0}
        assert state.journal.cache == {}

    async def test_agent_budget_pre_check_zero(self, tmp_path: Path) -> None:
        runner = MockWorkflowRunner()
        state = _state(tmp_path, runner, budget=0)
        with pytest.raises(WorkflowInputError, match="budget exceeded"):
            await state.agent("do x", label="x")
        assert runner.calls == []

    async def test_agent_cap(self, tmp_path: Path) -> None:
        runner = MockWorkflowRunner()
        state = _state(tmp_path, runner, agent_cap=1)
        await state.agent("a", label="a")
        with pytest.raises(WorkflowInputError, match="cap reached"):
            await state.agent("b", label="b")

    async def test_phase_dedup_and_log(self, tmp_path: Path) -> None:
        state = _state(tmp_path, MockWorkflowRunner())
        state.phase("A")
        state.phase("A")
        state.log("hello")
        types = [e["type"] for e in state.task.progress]
        assert types == ["workflow_phase", "workflow_log"]

    async def test_parallel_success(self, tmp_path: Path) -> None:
        state = _state(tmp_path, MockWorkflowRunner())
        results = await state.parallel([
            lambda: state.agent("a", label="a"),
            lambda: state.agent("b", label="b"),
        ])
        assert [r["label"] for r in results] == ["a", "b"]

    async def test_parallel_fail_fast(self, tmp_path: Path) -> None:
        def handler(prompt, schema, label):
            if label == "bad":
                raise WorkflowInputError("boom")
            return RunnerOutput(value=label, tokens=1)

        state = _state(tmp_path, MockWorkflowRunner(handler))
        with pytest.raises(WorkflowInputError, match="boom"):
            await state.parallel([
                lambda: state.agent("ok", label="ok"),
                lambda: state.agent("bad", label="bad"),
            ])

    async def test_pipeline_stages(self, tmp_path: Path) -> None:
        state = _state(tmp_path, MockWorkflowRunner())
        order: list[str] = []

        async def stage1(value, item, idx):
            order.append(f"s1:{item}")
            await asyncio.sleep(0.05 if idx == 0 else 0)
            return value + 1

        async def stage2(value, item, idx):
            order.append(f"s2:{item}")
            return value * 10

        results = await state.pipeline([1, 2], stage1, stage2)
        assert results == [20, 30]
        assert order.index("s2:2") < order.index("s2:1")

    async def test_workflow_nesting_one_level(self, tmp_path: Path) -> None:
        async def child(state: ExecutionState, args: dict[str, Any]) -> Any:
            with pytest.raises(WorkflowInputError, match="one level"):
                await state.workflow("demo-wf", {})
            return "child-done"

        workflows = {
            "demo-wf": (META, _echo_workflow),
            "child-wf": ({"name": "child-wf", "description": "c"}, child),
        }
        state = _state(tmp_path, MockWorkflowRunner(), workflows=workflows)
        result = await state.workflow("child-wf", {})
        assert result == "child-done"

    async def test_workflow_unknown_name(self, tmp_path: Path) -> None:
        state = _state(tmp_path, MockWorkflowRunner())
        with pytest.raises(WorkflowInputError, match="unknown workflow"):
            await state.workflow("nope", {})

    async def test_nested_shares_budget_and_limits(
        self, tmp_path: Path
    ) -> None:
        async def child(state: ExecutionState, args: dict[str, Any]) -> Any:
            return await state.agent("inner", label="inner")

        workflows = {
            "child-wf": ({"name": "child-wf", "description": "c"}, child),
        }
        state = _state(tmp_path, MockWorkflowRunner(), workflows=workflows)
        await state.workflow("child-wf", {})
        assert state.task.usage["agents"] == 1


# -- SubagentWorkflowRunner --


class FakeSubagentRunner:
    """按序返回 summary 的假 SubagentRunner。"""

    def __init__(self, summaries: list[str]) -> None:
        self._summaries = list(summaries)
        self.calls: list[str] = []

    async def run(self, spec: Any, task: str, parent: ToolContext) -> Any:
        self.calls.append(task)
        return SubagentResult(
            name=spec.name,
            summary=self._summaries.pop(0),
            usage=Usage(input_tokens=10, output_tokens=5),
        )


class TestSubagentWorkflowRunner:
    async def test_plain_text(self, tmp_path: Path) -> None:
        runner = SubagentWorkflowRunner(FakeSubagentRunner(["just text"]), "m")
        out = await runner.run("p", None, "l", _ctx(tmp_path))
        assert out.value == "just text"
        assert out.tokens == 15

    async def test_schema_valid(self, tmp_path: Path) -> None:
        schema = {"type": "object", "required": ["x"]}
        runner = SubagentWorkflowRunner(
            FakeSubagentRunner(['{"x": 1}']), "m"
        )
        out = await runner.run("p", schema, "l", _ctx(tmp_path))
        assert out.value == {"x": 1}

    async def test_schema_retry_once_then_ok(self, tmp_path: Path) -> None:
        schema = {"type": "object", "required": ["x"]}
        fake = FakeSubagentRunner(["not json", '{"x": 2}'])
        runner = SubagentWorkflowRunner(fake, "m")
        out = await runner.run("p", schema, "l", _ctx(tmp_path))
        assert out.value == {"x": 2}
        assert len(fake.calls) == 2
        assert "Return valid JSON" in fake.calls[1]
        assert out.tokens == 30

    async def test_schema_retry_still_invalid(self, tmp_path: Path) -> None:
        schema = {"type": "object", "required": ["x"]}
        runner = SubagentWorkflowRunner(
            FakeSubagentRunner(["bad", "still bad"]), "m"
        )
        with pytest.raises(WorkflowInputError, match="invalid output"):
            await runner.run("p", schema, "l", _ctx(tmp_path))

    async def test_schema_prompt_contains_schema(self, tmp_path: Path) -> None:
        schema = {"type": "object", "required": ["x"]}
        fake = FakeSubagentRunner(['{"x": 1}'])
        runner = SubagentWorkflowRunner(fake, "m")
        await runner.run("p", schema, "l", _ctx(tmp_path))
        assert '"required"' in fake.calls[0]


# -- WorkflowTool 端到端 --


class TestWorkflowTool:
    def test_tool_schema(self, tmp_path: Path) -> None:
        tool = _tool(tmp_path, MockWorkflowRunner())
        assert tool.name == "workflow"
        schema = tool.input_schema()
        assert schema["required"] == ["name"]
        assert schema["additionalProperties"] is False

    async def test_run_completes_with_artifacts(
        self, tmp_path: Path
    ) -> None:
        tool = _tool(tmp_path, MockWorkflowRunner())
        out = await tool.execute(
            {"name": "demo-wf", "args": {"x": "hi"}}, _ctx(tmp_path)
        )
        assert not out.is_error
        payload = json.loads(out.content)
        run_id = payload["launched"]["runId"]
        assert payload["result"] == {"label": "echo", "ok": True}
        assert payload["task"]["status"] == "completed"
        store = tmp_path / ".workflows"
        snapshot = _read_snapshot(store, run_id)
        assert snapshot["workflowName"] == "demo-wf"
        assert (store / f"{run_id}.output.json").exists()
        assert (store / f"{run_id}.journal.jsonl").exists()
        assert (store / "last_run.txt").read_text().strip() == run_id

    async def test_unknown_name(self, tmp_path: Path) -> None:
        tool = _tool(tmp_path, MockWorkflowRunner())
        out = await tool.execute({"name": "nope"}, _ctx(tmp_path))
        assert out.is_error
        assert "unknown workflow" in out.content

    async def test_failed_run_persists_status(self, tmp_path: Path) -> None:
        async def boom(state: ExecutionState, args: dict[str, Any]) -> Any:
            raise RuntimeError("kaboom")

        tool = WorkflowTool(
            MockWorkflowRunner(),
            tmp_path / ".workflows",
            workflows={"demo-wf": (META, boom)},
        )
        out = await tool.execute({"name": "demo-wf"}, _ctx(tmp_path))
        payload = json.loads(out.content)
        assert payload["task"]["status"] == "failed"
        assert payload["result"] == {"error": "kaboom"}

    async def test_resume_replays_cached_agents(self, tmp_path: Path) -> None:
        calls: list[str] = []

        async def two_agents(state: ExecutionState, args: dict[str, Any]) -> Any:
            await state.agent("first", label="first")
            if not calls:
                calls.append("failed-once")
                raise RuntimeError("crash after first agent")
            await state.agent("second", label="second")
            return "done"

        runner = MockWorkflowRunner()
        workflows = {"demo-wf": (META, two_agents)}
        tool = WorkflowTool(runner, tmp_path / ".workflows", workflows=workflows)
        ctx = _ctx(tmp_path)
        first = json.loads((await tool.execute({"name": "demo-wf"}, ctx)).content)
        assert first["task"]["status"] == "failed"
        assert len(runner.calls) == 1

        runner2 = MockWorkflowRunner()
        tool2 = WorkflowTool(runner2, tmp_path / ".workflows", workflows=workflows)
        resumed = json.loads(
            (
                await tool2.execute(
                    {"name": "demo-wf",
                     "resume_from_run_id": first["launched"]["runId"]},
                    ctx,
                )
            ).content
        )
        assert resumed["task"]["status"] == "completed"
        assert resumed["result"] == "done"
        assert len(runner2.calls) == 1
        assert runner2.calls[0][2] == "second"

    async def test_resume_args_mismatch(self, tmp_path: Path) -> None:
        tool = _tool(tmp_path, MockWorkflowRunner())
        ctx = _ctx(tmp_path)
        first = json.loads(
            (await tool.execute({"name": "demo-wf", "args": {"x": "1"}}, ctx)).content
        )
        out = await tool.execute(
            {"name": "demo-wf", "args": {"x": "2"},
             "resume_from_run_id": first["launched"]["runId"]},
            ctx,
        )
        assert out.is_error
        assert "args do not match" in out.content

    async def test_resume_workflow_name_mismatch(self, tmp_path: Path) -> None:
        other_meta = {"name": "other-wf", "description": "o"}
        tool = _tool(tmp_path, MockWorkflowRunner())
        ctx = _ctx(tmp_path)
        first = json.loads((await tool.execute({"name": "demo-wf"}, ctx)).content)
        tool2 = WorkflowTool(
            MockWorkflowRunner(),
            tmp_path / ".workflows",
            workflows={"other-wf": (other_meta, _echo_workflow)},
        )
        out = await tool2.execute(
            {"name": "other-wf",
             "resume_from_run_id": first["launched"]["runId"]},
            ctx,
        )
        assert out.is_error
        assert "does not match workflow meta" in out.content

    async def test_resume_without_args_uses_saved(self, tmp_path: Path) -> None:
        captured: list[dict] = []

        async def capture(state: ExecutionState, args: dict[str, Any]) -> Any:
            captured.append(dict(args))
            return "ok"

        tool = WorkflowTool(
            MockWorkflowRunner(),
            tmp_path / ".workflows",
            workflows={"demo-wf": (META, capture)},
        )
        ctx = _ctx(tmp_path)
        first = json.loads(
            (await tool.execute({"name": "demo-wf", "args": {"x": "1"}}, ctx)).content
        )
        await tool.execute(
            {"name": "demo-wf",
             "resume_from_run_id": first["launched"]["runId"]},
            ctx,
        )
        assert captured == [{"x": "1"}, {"x": "1"}]

    async def test_run_lock_serializes(self, tmp_path: Path) -> None:
        release = asyncio.Event()
        order: list[str] = []

        async def slow(state: ExecutionState, args: dict[str, Any]) -> Any:
            order.append("start")
            await release.wait()
            order.append("end")
            return "ok"

        tool = WorkflowTool(
            MockWorkflowRunner(),
            tmp_path / ".workflows",
            workflows={"demo-wf": (META, slow)},
        )
        ctx = _ctx(tmp_path)
        t1 = asyncio.create_task(tool.execute({"name": "demo-wf"}, ctx))
        await asyncio.sleep(0.05)
        release.set()
        out = await t1
        assert order == ["start", "end"]
        assert not out.is_error

    async def test_event_sink_receives_notices(self, tmp_path: Path) -> None:
        sink: asyncio.Queue = asyncio.Queue()
        tool = _tool(tmp_path, MockWorkflowRunner(), event_sink=sink)
        await tool.execute({"name": "demo-wf"}, _ctx(tmp_path))
        texts = []
        while not sink.empty():
            texts.append(sink.get_nowait().text)
        assert any("agent done" in t for t in texts)
        assert any("demo-wf completed" in t for t in texts)


# -- register_workflow --


class TestRegistry:
    def test_register_and_lookup(self) -> None:
        from mcc_harness import WORKFLOWS

        async def fn(state: ExecutionState, args: dict[str, Any]) -> Any:
            return "ok"

        register_workflow(
            {"name": "test-registry-wf", "description": "d"}, fn
        )
        try:
            assert "test-registry-wf" in WORKFLOWS
            assert WORKFLOWS["test-registry-wf"][1] is fn
        finally:
            WORKFLOWS.pop("test-registry-wf", None)

    def test_register_validates_meta(self) -> None:
        async def fn(state: ExecutionState, args: dict[str, Any]) -> Any:
            return None

        with pytest.raises(WorkflowInputError):
            register_workflow({"name": "Bad Name", "description": "d"}, fn)


# -- Config --


class TestWorkflowConfig:
    def test_defaults(self) -> None:
        cfg = WorkflowConfig()
        assert cfg.enabled is True
        assert cfg.concurrency == 4
        assert cfg.agent_cap == 20
        assert cfg.budget_tokens is None

    def test_config_default_in_root(self) -> None:
        assert Config().workflow == WorkflowConfig()

    def test_merge_project_overrides(self) -> None:
        home = Config()
        project = Config(workflow=WorkflowConfig(agent_cap=5))
        merged = merge(home, project)
        assert merged.workflow.agent_cap == 5

    def test_merge_project_default_keeps_home(self) -> None:
        home = Config(workflow=WorkflowConfig(concurrency=8))
        project = Config()
        assert merge(home, project).workflow.concurrency == 8
