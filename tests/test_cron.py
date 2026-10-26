"""cron 调度器测试：表达式解析 / CronScheduler / 工具 / AgentLoop 注入。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from mcc_config.loader import merge
from mcc_config.models import Config, CronConfig
from mcc_core import Notice, ToolContext
from mcc_harness import (
    AgentLoopBuilder,
    CancelCronTool,
    CronScheduler,
    ListCronsTool,
    ScheduleCronTool,
    cron_matches,
    format_cron_delivery,
    validate_cron,
)
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


# 2026-06-10 为周三，cron weekday 编码 0=周日，故周三=3
def _moment(minute: int, hour: int = 9, day: int = 10, month: int = 6) -> datetime:
    return datetime(2026, month, day, hour, minute)


# -- cron 表达式匹配 --


class TestCronMatches:
    @pytest.mark.parametrize(
        ("expr", "moment", "expected"),
        [
            ("* * * * *", _moment(30), True),
            ("0 9 * * *", _moment(0, 9), True),
            ("0 9 * * *", _moment(1, 9), False),
            ("30 * * * *", _moment(30, 10), True),
            ("*/15 * * * *", _moment(0), True),
            ("*/15 * * * *", _moment(15), True),
            ("*/15 * * * *", _moment(7), False),
            ("0-30 * * * *", _moment(25), True),
            ("0-30 * * * *", _moment(31), False),
            ("0,30 * * * *", _moment(0), True),
            ("0,30 * * * *", _moment(30), True),
            ("0,30 * * * *", _moment(15), False),
            ("* 9-17 * * *", _moment(0, 12), True),
            ("* 9-17 * * *", _moment(0, 18), False),
            ("* * * 6 *", _moment(0, 9, day=15, month=6), True),
            ("* * * 3 *", _moment(0, 9, day=15, month=6), False),
            ("* * * * 3", _moment(0, 9, day=10), True),
            ("* * * * 1", _moment(0, 9, day=10), False),
            ("* * * * 0", _moment(0, 9, day=10), False),
            ("* * 15 * 1", _moment(0, 9, day=15), True),
            # 2026-06-15 是周一：day 不匹配但 weekday=1 命中（OR 语义）
            ("* * 10 * 1", _moment(0, 9, day=15), True),
            # 2026-06-16 是周二：day 与 weekday 均不匹配
            ("* * 10 * 1", _moment(0, 9, day=16), False),
            ("* * 15 * 1", _moment(0, 9, day=16), False),
            ("bad expr", _moment(0), False),
        ],
    )
    def test_match(self, expr: str, moment: datetime, expected: bool) -> None:
        assert cron_matches(expr, moment) is expected

    def test_weekday_conversion(self) -> None:
        # 2026-06-10 是周三：cron weekday 3 匹配、1（周一）不匹配
        assert cron_matches("* * * * 3", _moment(0, 9, day=10)) is True
        assert cron_matches("* * * * 1", _moment(0, 9, day=10)) is False


class TestValidateCron:
    @pytest.mark.parametrize(
        "expr",
        [
            "* * * * *",
            "0 9 * * *",
            "*/15 9-17 1,15 3 0-6",
            "59 23 31 12 6",
        ],
    )
    def test_valid(self, expr: str) -> None:
        assert validate_cron(expr) is None

    @pytest.mark.parametrize(
        ("expr", "fragment"),
        [
            ("* * * *", "Expected 5 fields"),
            ("* * * * * *", "Expected 5 fields"),
            ("60 * * * *", "outside [0-59]"),
            ("* 24 * * *", "outside [0-23]"),
            ("* * 0 * *", "outside [1-31]"),
            ("* * * 13 *", "outside [1-12]"),
            ("* * * * 7", "outside [0-6]"),
            ("*/0 * * * *", "Invalid step"),
            ("*/x * * * *", "Invalid step"),
            ("5-3 * * * *", "greater than end"),
            ("a * * * *", "Invalid field"),
        ],
    )
    def test_invalid(self, expr: str, fragment: str) -> None:
        error = validate_cron(expr)
        assert error is not None
        assert fragment in error


# -- CronScheduler --


class TestCronScheduler:
    async def test_schedule_and_list(self, tmp_path: Path) -> None:
        sched = CronScheduler()
        job = await sched.schedule("0 9 * * *", "run tests")
        assert isinstance(job.id, str)
        assert job.id.startswith("cron_")
        jobs = await sched.list_jobs()
        assert [j.id for j in jobs] == [job.id]

    async def test_schedule_invalid_cron(self) -> None:
        sched = CronScheduler()
        result = await sched.schedule("bad", "x")
        assert isinstance(result, str)
        assert "Expected 5 fields" in result

    async def test_schedule_empty_prompt(self) -> None:
        sched = CronScheduler()
        result = await sched.schedule("* * * * *", "  ")
        assert result == "Prompt cannot be empty"

    async def test_poll_fires_due_job(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run tests")
        fired = await sched.poll(datetime.now())
        assert fired == 1
        jobs = await sched.consume()
        assert len(jobs) == 1
        assert jobs[0].prompt == "run tests"
        assert await sched.consume() == []

    async def test_poll_same_minute_not_refired(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run tests")
        moment = datetime.now()
        assert await sched.poll(moment) == 1
        assert await sched.poll(moment) == 0

    async def test_poll_next_minute_refires_recurring(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run tests")
        moment = datetime.now()
        await sched.poll(moment)
        jobs = await sched.consume()
        await sched.acknowledge(jobs)
        next_minute = datetime.now().replace(second=0, microsecond=0)
        from datetime import timedelta

        assert await sched.poll(next_minute + timedelta(minutes=1)) == 1

    async def test_poll_pending_delivery_skips(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run tests")
        moment = datetime.now()
        await sched.poll(moment)
        from datetime import timedelta

        assert await sched.poll(moment + timedelta(minutes=1)) == 0

    async def test_non_matching_expr_never_fires(self) -> None:
        sched = CronScheduler()
        await sched.schedule("0 0 1 1 *", "happy new year")
        moment = datetime(2026, 6, 10, 9, 30)
        assert await sched.poll(moment) == 0

    async def test_acknowledge_recurring_resets(self) -> None:
        sched = CronScheduler()
        job = await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())
        jobs = await sched.consume()
        await sched.acknowledge(jobs)
        remaining = await sched.list_jobs()
        assert [j.id for j in remaining] == [job.id]
        assert remaining[0].pending_delivery is False

    async def test_acknowledge_one_shot_removes(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "x", recurring=False)
        await sched.poll(datetime.now())
        jobs = await sched.consume()
        await sched.acknowledge(jobs)
        assert await sched.list_jobs() == []

    async def test_restore_requeues(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())
        jobs = await sched.consume()
        await sched.restore(jobs)
        requeued = await sched.consume()
        assert [j.id for j in requeued] == [j.id for j in jobs]
        assert requeued[0].pending_delivery is True

    async def test_restore_deduplicates(self) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())
        jobs = await sched.consume()
        await sched.restore(jobs)
        await sched.restore(jobs)
        assert len(await sched.consume()) == 1

    async def test_cancel_removes_and_drains_queue(self) -> None:
        sched = CronScheduler()
        job = await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())
        result = await sched.cancel(job.id)
        assert result == f"Cancelled {job.id}"
        assert await sched.list_jobs() == []
        assert await sched.consume() == []

    async def test_cancel_unknown_id(self) -> None:
        sched = CronScheduler()
        result = await sched.cancel("cron_nope")
        assert "not found" in result

    async def test_start_stop_lifecycle(self) -> None:
        sched = CronScheduler(poll_interval=0.05)
        await sched.start()
        try:
            await sched.schedule("* * * * *", "x")
            await asyncio.sleep(0.2)
            jobs = await sched.consume()
            assert len(jobs) == 1
        finally:
            await sched.stop()

    async def test_start_idempotent(self) -> None:
        sched = CronScheduler()
        await sched.start()
        first = sched._poll_task
        await sched.start()
        assert sched._poll_task is first
        await sched.stop()

    async def test_stop_twice_safe(self) -> None:
        sched = CronScheduler()
        await sched.start()
        await sched.stop()
        await sched.stop()


# -- durable 持久化 --


class TestDurablePersistence:
    async def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        sched = CronScheduler(durable_path=path)
        job = await sched.schedule("0 9 * * *", "run tests")

        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload[0]["id"] == job.id

        sched2 = CronScheduler(durable_path=path)
        await sched2.start()
        try:
            jobs = await sched2.list_jobs()
            assert [j.id for j in jobs] == [job.id]
            assert jobs[0].cron == "0 9 * * *"
            assert jobs[0].pending_delivery is False
        finally:
            await sched2.stop()

    async def test_session_job_not_persisted(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        sched = CronScheduler(durable_path=path)
        await sched.schedule("0 9 * * *", "x", durable=False)
        assert not path.exists()

    async def test_reload_pending_delivery_requeues(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        sched = CronScheduler(durable_path=path)
        await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())

        sched2 = CronScheduler(durable_path=path)
        await sched2.start()
        try:
            jobs = await sched2.consume()
            assert len(jobs) == 1
        finally:
            await sched2.stop()

    async def test_reload_skips_invalid_entries(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "cron_bad", "cron": "not cron", "prompt": "x"},
                    {"id": "cron_ok", "cron": "0 9 * * *", "prompt": "ok"},
                    {"id": "cron_empty", "cron": "* * * * *", "prompt": "  "},
                    {"id": "bad_prefix", "cron": "* * * * *", "prompt": "x"},
                ]
            ),
            encoding="utf-8",
        )
        sched = CronScheduler(durable_path=path)
        await sched.start()
        try:
            jobs = await sched.list_jobs()
            assert [j.id for j in jobs] == ["cron_ok"]
        finally:
            await sched.stop()

    async def test_reload_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        path.write_text("not json", encoding="utf-8")
        sched = CronScheduler(durable_path=path)
        await sched.start()
        try:
            assert await sched.list_jobs() == []
        finally:
            await sched.stop()

    async def test_one_shot_removed_after_ack(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        sched = CronScheduler(durable_path=path)
        await sched.schedule("* * * * *", "x", recurring=False)
        await sched.poll(datetime.now())
        jobs = await sched.consume()
        await sched.acknowledge(jobs)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == []

    async def test_cancel_persists(self, tmp_path: Path) -> None:
        path = tmp_path / ".cron_jobs.json"
        sched = CronScheduler(durable_path=path)
        job = await sched.schedule("0 9 * * *", "x")
        await sched.cancel(job.id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == []


# -- format_cron_delivery --


class TestFormatCronDelivery:
    def test_format(self) -> None:
        from mcc_harness.cron import CronJob

        job = CronJob(id="cron_ab12", cron="0 9 * * *", prompt="run tests")
        text = format_cron_delivery([job])
        assert text.startswith("<cron_delivery>")
        assert text.endswith("</cron_delivery>")
        assert 'id="cron_ab12"' in text
        assert 'cron="0 9 * * *"' in text
        assert "run tests" in text

    def test_format_empty(self) -> None:
        assert format_cron_delivery([]) == (
            "<cron_delivery>\n</cron_delivery>"
        )


# -- 工具 --


class TestCronTools:
    async def test_schedule_tool(self, tmp_path: Path) -> None:
        sched = CronScheduler()
        tool = ScheduleCronTool(sched)
        out = await tool.execute(
            {"cron": "0 9 * * *", "prompt": "run tests"}, _ctx(tmp_path)
        )
        assert out.content.startswith("Scheduled cron_")
        assert "0 9 * * *" in out.content
        assert len(await sched.list_jobs()) == 1

    async def test_schedule_tool_invalid(self, tmp_path: Path) -> None:
        tool = ScheduleCronTool(CronScheduler())
        out = await tool.execute({"cron": "bad", "prompt": "x"}, _ctx(tmp_path))
        assert out.content.startswith("Error:")

    async def test_list_tool_empty(self, tmp_path: Path) -> None:
        tool = ListCronsTool(CronScheduler())
        out = await tool.execute({}, _ctx(tmp_path))
        assert out.content == "No cron jobs."

    async def test_list_tool_with_jobs(self, tmp_path: Path) -> None:
        sched = CronScheduler()
        await sched.schedule("0 9 * * *", "run tests", recurring=False, durable=False)
        tool = ListCronsTool(sched)
        out = await tool.execute({}, _ctx(tmp_path))
        assert "cron_" in out.content
        assert "[one-shot, session]" in out.content

    async def test_cancel_tool(self, tmp_path: Path) -> None:
        sched = CronScheduler()
        job = await sched.schedule("0 9 * * *", "x")
        tool = CancelCronTool(sched)
        out = await tool.execute({"job_id": job.id}, _ctx(tmp_path))
        assert out.content == f"Cancelled {job.id}"

    async def test_cancel_tool_unknown(self, tmp_path: Path) -> None:
        tool = CancelCronTool(CronScheduler())
        out = await tool.execute({"job_id": "cron_nope"}, _ctx(tmp_path))
        assert "not found" in out.content

    def test_tool_schemas(self) -> None:
        tool = ScheduleCronTool(CronScheduler())
        schema = tool.input_schema()
        assert set(schema["required"]) == {"cron", "prompt"}
        assert tool.name == "schedule_cron"


# -- AgentLoop 注入 --


class _NoOpTool:
    """测试占位工具。"""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def description(self) -> str:
        return "noop"

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> Any:
        from mcc_core import ToolOutput

        return ToolOutput.ok("ok")


class TestAgentLoopInjection:
    def _make_agent(self, llm, ctx, sched):
        registry = ToolRegistry()
        registry.register(_NoOpTool())
        return (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .cron(sched)
            .build()
        )

    async def test_due_job_injected_as_message(self, ctx: ToolContext) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run the test suite")
        await sched.poll(datetime.now())
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sched)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        cron_msgs = [
            m for m in agent.history
            if m.role.value == "user"
            and any(
                "<cron_delivery>" in getattr(b, "text", "")
                for b in m.content
            )
        ]
        assert len(cron_msgs) == 1

    async def test_due_job_emits_notice(self, ctx: ToolContext) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "run the test suite")
        await sched.poll(datetime.now())
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sched)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        notices = []
        while not queue.empty():
            ev = queue.get_nowait()
            if isinstance(ev, Notice):
                notices.append(ev)
        assert any("<cron_delivery>" in n.text for n in notices)

    async def test_one_shot_acknowledged_after_injection(
        self, ctx: ToolContext
    ) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "x", recurring=False)
        await sched.poll(datetime.now())
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sched)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert await sched.list_jobs() == []

    async def test_recurring_survives_injection(self, ctx: ToolContext) -> None:
        sched = CronScheduler()
        await sched.schedule("* * * * *", "x")
        await sched.poll(datetime.now())
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, sched)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        jobs = await sched.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].pending_delivery is False

    async def test_no_scheduler_no_injection(self, ctx: ToolContext) -> None:
        llm = MockLlm([single_text_turn("done")])
        agent = self._make_agent(llm, ctx, None)
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert not any(
            "<cron_delivery>" in getattr(b, "text", "")
            for m in agent.history
            for b in m.content
        )


# -- Config --


class TestCronConfig:
    def test_defaults(self) -> None:
        cfg = CronConfig()
        assert cfg.enabled is True
        assert cfg.poll_interval == 1.0
        assert cfg.durable is True

    def test_config_default_in_root(self) -> None:
        assert Config().cron == CronConfig()

    def test_merge_project_overrides(self) -> None:
        home = Config()
        project = Config(cron=CronConfig(enabled=False, poll_interval=5.0))
        merged = merge(home, project)
        assert merged.cron.enabled is False
        assert merged.cron.poll_interval == 5.0

    def test_merge_project_default_keeps_home(self) -> None:
        home = Config(cron=CronConfig(poll_interval=2.0))
        project = Config()
        merged = merge(home, project)
        assert merged.cron.poll_interval == 2.0
