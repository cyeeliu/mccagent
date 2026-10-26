"""Cron 调度器：按 cron 表达式定时触发 prompt，注入 AgentLoop。

调度器在 asyncio.Task 中每秒轮询 due jobs，匹配的 job 入队
（asyncio.Queue）；AgentLoop 每轮迭代首 consume 并注入为 user
消息，使模型在空闲时处理定时任务。durable job 持久化到
``.cron_jobs.json``，进程重启后自动恢复。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcc_core import Tool, ToolContext, ToolOutput

# -- Cron expression parsing --


def _matches_field(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) == 0:
            return False
        return value % int(step) == 0
    if "," in field:
        return any(_matches_field(part.strip(), value) for part in field.split(","))
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return False
        return int(start) <= value <= int(end)
    if not field.isdigit():
        return False
    return value == int(field)


def cron_matches(cron_expr: str, moment: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, day, month, weekday = fields
    cron_weekday = (moment.weekday() + 1) % 7
    if not (
        _matches_field(minute, moment.minute)
        and _matches_field(hour, moment.hour)
        and _matches_field(month, moment.month)
    ):
        return False
    day_matches = _matches_field(day, moment.day)
    weekday_matches = _matches_field(weekday, cron_weekday)
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches


def _validate_field(field: str, minimum: int, maximum: int) -> str | None:
    if field == "*":
        return None
    if field.startswith("*/"):
        step = field[2:]
        if not step.isdigit() or int(step) <= 0:
            return f"Invalid step: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            error = _validate_field(part.strip(), minimum, maximum)
            if error:
                return error
        return None
    if "-" in field:
        start, end = field.split("-", 1)
        if not start.isdigit() or not end.isdigit():
            return f"Invalid range: {field}"
        start_value, end_value = int(start), int(end)
        if start_value > end_value:
            return f"Range start is greater than end: {field}"
        if start_value < minimum or end_value > maximum:
            return f"Range {field} is outside [{minimum}-{maximum}]"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    value = int(field)
    if value < minimum or value > maximum:
        return f"Value {value} is outside [{minimum}-{maximum}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    field_rules = [
        ("minute", 0, 59),
        ("hour", 0, 23),
        ("day-of-month", 1, 31),
        ("month", 1, 12),
        ("day-of-week", 0, 6),
    ]
    for field_value, (name, minimum, maximum) in zip(
        fields, field_rules, strict=True
    ):
        error = _validate_field(field_value, minimum, maximum)
        if error:
            return f"{name}: {error}"
    return None


# -- CronJob --


@dataclass
class CronJob:
    """定时任务定义。"""

    id: str
    cron: str
    prompt: str
    recurring: bool = True
    durable: bool = True
    pending_delivery: bool = False
    last_fired: str | None = None
    created_at: str = ""


# -- CronScheduler --


class CronScheduler:
    """asyncio 驱动的 cron 调度器。

    ``start`` 启动 poll loop（asyncio.Task），每 ``poll_interval``
    秒检查 due jobs 并入队；``consume`` 由 AgentLoop 每轮迭代首
    调用，取走 due jobs；``acknowledge`` 标记投递成功，
    ``restore`` 在投递失败时恢复队列。
    """

    def __init__(
        self,
        durable_path: Path | None = None,
        poll_interval: float = 1.0,
    ) -> None:
        self._jobs: dict[str, CronJob] = {}
        self._queue: asyncio.Queue[CronJob] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._durable_path = durable_path
        self._poll_interval = poll_interval
        self._poll_task: asyncio.Task[None] | None = None
        self._stopped = False

    # -- lifecycle --

    async def start(self) -> None:
        if self._poll_task is not None:
            return
        self._stopped = False
        if self._durable_path is not None:
            self._load_durable()
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

    async def _poll_loop(self) -> None:
        while not self._stopped:
            with contextlib.suppress(Exception):
                await self.poll(datetime.now())
            await asyncio.sleep(self._poll_interval)

    # -- scheduling --

    async def schedule(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
    ) -> CronJob | str:
        error = validate_cron(cron)
        if error:
            return error
        if not prompt.strip():
            return "Prompt cannot be empty"
        async with self._lock:
            job = CronJob(
                id=self._new_id(),
                cron=cron,
                prompt=prompt,
                recurring=recurring,
                durable=durable,
                created_at=datetime.now().isoformat(),
            )
            self._jobs[job.id] = job
            try:
                if durable:
                    self._save_durable()
            except Exception:
                self._jobs.pop(job.id, None)
                raise
        return job

    async def cancel(self, job_id: str) -> str:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return f"Job {job_id} not found"
            self._jobs.pop(job_id)
            self._drain_queue(job_id)
            try:
                if job.durable:
                    self._save_durable()
            except Exception:
                self._jobs[job_id] = job
                raise
        return f"Cancelled {job_id}"

    async def list_jobs(self) -> list[CronJob]:
        async with self._lock:
            return list(self._jobs.values())

    # -- poll / consume / acknowledge --

    async def poll(self, moment: datetime) -> int:
        minute_marker = moment.strftime("%Y-%m-%d %H:%M")
        fired = 0
        async with self._lock:
            for job in list(self._jobs.values()):
                if job.pending_delivery or job.last_fired == minute_marker:
                    continue
                if cron_matches(job.cron, moment):
                    job.pending_delivery = True
                    job.last_fired = minute_marker
                    try:
                        if job.durable:
                            self._save_durable()
                    except Exception:
                        job.pending_delivery = False
                        job.last_fired = None
                        raise
                    await self._queue.put(job)
                    fired += 1
        return fired

    async def consume(self) -> list[CronJob]:
        jobs: list[CronJob] = []
        while not self._queue.empty():
            jobs.append(self._queue.get_nowait())
        return jobs

    async def acknowledge(self, jobs: list[CronJob]) -> None:
        async with self._lock:
            for delivered in jobs:
                current = self._jobs.get(delivered.id)
                if current is None:
                    continue
                if current.recurring:
                    current.pending_delivery = False
                else:
                    self._jobs.pop(current.id)
            if any(j.durable for j in jobs):
                try:
                    self._save_durable()
                except Exception:
                    raise

    async def restore(self, jobs: list[CronJob]) -> None:
        async with self._lock:
            queued = self._drain_all()
            queued_ids = {j.id for j in queued}
            for delivered in jobs:
                current = self._jobs.get(delivered.id)
                if current is None:
                    continue
                current.pending_delivery = True
                if current.id not in queued_ids:
                    queued.append(current)
                    queued_ids.add(current.id)
            for job in queued:
                await self._queue.put(job)

    # -- durable persistence --

    def _save_durable(self) -> None:
        if self._durable_path is None:
            return
        payload = [asdict(job) for job in self._jobs.values() if job.durable]
        tmp = self._durable_path.with_name(
            f"{self._durable_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._durable_path)
        finally:
            tmp.unlink(missing_ok=True)

    def _load_durable(self) -> int:
        if self._durable_path is None or not self._durable_path.exists():
            return 0
        try:
            payload = json.loads(self._durable_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return 0
        except (OSError, json.JSONDecodeError):
            return 0
        loaded = 0
        for item in payload:
            try:
                job = CronJob(**item)
                error = validate_cron(job.cron)
                if error:
                    continue
                if not job.id.startswith("cron_"):
                    continue
                if not job.prompt.strip():
                    continue
            except (TypeError, ValueError):
                continue
            self._jobs[job.id] = job
            if job.pending_delivery:
                self._queue.put_nowait(job)
            loaded += 1
        return loaded

    # -- helpers --

    def _new_id(self) -> str:
        for _ in range(100):
            job_id = f"cron_{secrets.token_hex(4)}"
            if job_id not in self._jobs:
                return job_id
        raise RuntimeError("Could not allocate a cron job ID")

    def _drain_queue(self, job_id: str) -> None:
        remaining: list[CronJob] = []
        while not self._queue.empty():
            job = self._queue.get_nowait()
            if job.id != job_id:
                remaining.append(job)
        for job in remaining:
            self._queue.put_nowait(job)

    def _drain_all(self) -> list[CronJob]:
        jobs: list[CronJob] = []
        while not self._queue.empty():
            jobs.append(self._queue.get_nowait())
        return jobs


# -- formatting --


def format_cron_delivery(jobs: list[CronJob]) -> str:
    """将 due cron jobs 格式化为注入消息。"""
    lines = ["<cron_delivery>"]
    for job in jobs:
        lines.append(f"  <job id=\"{job.id}\" cron=\"{job.cron}\">")
        lines.append(f"    {job.prompt}")
        lines.append("  </job>")
    lines.append("</cron_delivery>")
    return "\n".join(lines)


# -- Tools --


class ScheduleCronTool(Tool):
    """schedule_cron：按 5 字段 cron 表达式调度 prompt。"""

    def __init__(self, scheduler: CronScheduler) -> None:
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        return "schedule_cron"

    @property
    def description(self) -> str:
        return (
            "Schedule a prompt to run on a cron schedule. "
            "Uses standard 5-field cron: minute hour day-of-month month day-of-week. "
            "The prompt will be injected into the agent loop at the scheduled time."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cron": {
                    "type": "string",
                    "description": "5-field cron expression, e.g. '0 9 * * *' for daily at 9am",
                },
                "prompt": {"type": "string", "description": "Prompt to run at the scheduled time"},
                "recurring": {
                    "type": "boolean",
                    "description": "If true, repeats on schedule; if false, runs once then removed",
                },
                "durable": {
                    "type": "boolean",
                    "description": "If true, persists to disk and survives restart",
                },
            },
            "required": ["cron", "prompt"],
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        cron = input["cron"]
        prompt = input["prompt"]
        recurring = input.get("recurring", True)
        durable = input.get("durable", True)
        result = await self._scheduler.schedule(cron, prompt, recurring, durable)
        if isinstance(result, str):
            return ToolOutput.err(f"Error: {result}")
        return ToolOutput.ok(f"Scheduled {result.id}: {cron} -> {prompt}")


class ListCronsTool(Tool):
    """list_crons：列出所有定时任务。"""

    def __init__(self, scheduler: CronScheduler) -> None:
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        return "list_crons"

    @property
    def description(self) -> str:
        return "List all scheduled cron jobs."

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        jobs = await self._scheduler.list_jobs()
        if not jobs:
            return ToolOutput.ok("No cron jobs.")
        lines = []
        for job in jobs:
            frequency = "recurring" if job.recurring else "one-shot"
            storage = "durable" if job.durable else "session"
            lines.append(
                f"{job.id}: {job.cron} -> {job.prompt[:60]} [{frequency}, {storage}]"
            )
        return ToolOutput.ok("\n".join(lines))


class CancelCronTool(Tool):
    """cancel_cron：按 ID 取消定时任务。"""

    def __init__(self, scheduler: CronScheduler) -> None:
        self._scheduler = scheduler

    @property
    def name(self) -> str:
        return "cancel_cron"

    @property
    def description(self) -> str:
        return "Cancel a scheduled cron job by its ID."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        job_id = input["job_id"]
        result = await self._scheduler.cancel(job_id)
        return ToolOutput.ok(result)
