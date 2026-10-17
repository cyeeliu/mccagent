"""会话持久化测试：录制、恢复、列表、清理与标题生成。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mcc_core import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_session import (
    SessionMeta,
    SessionNotFoundError,
    SessionRecorder,
    TurnSnapshot,
    build_turn_snapshot,
    list_sessions,
    load_session,
    maybe_generate_title,
    purge_sessions,
)


def _snap(
    iteration: int = 1,
    request_messages: list[Message] | None = None,
    assistant_blocks: list | None = None,
    tool_outputs: list[tuple[str, str, bool]] | None = None,
    cost_usd: float = 0.01,
) -> TurnSnapshot:
    return build_turn_snapshot(
        iteration=iteration,
        request_messages=request_messages or [Message.user("hello")],
        assistant_blocks=assistant_blocks or [TextBlock(text="hi")],
        tool_outputs=tool_outputs or [],
        usage=Usage(input_tokens=100, output_tokens=20),
        model="claude-sonnet-4-6",
        cost_usd=cost_usd,
    )


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture
async def recorder(base: Path, tmp_path: Path) -> SessionRecorder:
    return await SessionRecorder.open("sess1", tmp_path, base_dir=base)


class TestSessionRecorder:
    @pytest.mark.asyncio
    async def test_open_creates_meta(self, base: Path, tmp_path: Path):
        r = await SessionRecorder.open("abc", tmp_path, base_dir=base)
        assert r.meta.id == "abc"
        assert r.meta.turns == 0
        assert r.meta.cost_usd == 0.0
        assert r.meta.status == "running"
        assert r.meta.cwd == str(tmp_path)
        day_dir = base / datetime.now(UTC).strftime("%Y-%m-%d")
        assert (day_dir / "abc.meta.json").exists()

    @pytest.mark.asyncio
    async def test_open_sync_matches_async(self, base: Path, tmp_path: Path):
        r = SessionRecorder.open_sync("sync1", tmp_path, base_dir=base)
        assert r.meta.id == "sync1"
        day_dir = base / datetime.now(UTC).strftime("%Y-%m-%d")
        assert (day_dir / "sync1.meta.json").exists()

    @pytest.mark.asyncio
    async def test_record_appends_jsonl_and_updates_meta(self, recorder):
        await recorder.record(_snap(cost_usd=0.01))
        await recorder.record(_snap(iteration=2, cost_usd=0.02))

        day_dir = recorder._meta_path.parent
        lines = (day_dir / "sess1.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        first = TurnSnapshot.model_validate_json(lines[0])
        assert first.iteration == 1
        assert first.model == "claude-sonnet-4-6"
        assert first.cost_usd == 0.01

        assert recorder.meta.turns == 2
        assert recorder.meta.cost_usd == pytest.approx(0.03)
        assert recorder.meta.status == "running"

    @pytest.mark.asyncio
    async def test_cost_accumulated_not_reestimated(self, recorder):
        """meta 成本只能来自 TurnSnapshot.cost_usd 的累加。"""
        snap = _snap(cost_usd=0.123)
        await recorder.record(snap)
        assert recorder.meta.cost_usd == pytest.approx(0.123)

    @pytest.mark.asyncio
    async def test_finalize_writes_status(self, recorder, base: Path):
        await recorder.record(_snap())
        await recorder.finalize("ended")

        meta_path = recorder._meta_path
        meta = SessionMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        assert meta.status == "ended"
        assert meta.turns == 1

    @pytest.mark.asyncio
    async def test_open_recovers_existing_meta(self, base: Path, tmp_path: Path):
        r1 = await SessionRecorder.open("keep", tmp_path, base_dir=base)
        await r1.record(_snap(cost_usd=0.05))
        r1.meta.title = "已有标题"
        await r1.save_meta()
        await r1.finalize("ended")

        r2 = await SessionRecorder.open("keep", tmp_path, base_dir=base)
        assert r2.meta.turns == 1
        assert r2.meta.cost_usd == pytest.approx(0.05)
        assert r2.meta.title == "已有标题"
        assert r2.meta.status == "running"

    @pytest.mark.asyncio
    async def test_save_meta_persists_title(self, recorder):
        recorder.meta.title = "新标题"
        await recorder.save_meta()
        meta = SessionMeta.model_validate_json(
            recorder._meta_path.read_text(encoding="utf-8")
        )
        assert meta.title == "新标题"

    @pytest.mark.asyncio
    async def test_snapshot_lists_are_copied(self):
        msgs = [Message.user("a")]
        blocks = [TextBlock(text="b")]
        outputs: list[tuple[str, str, bool]] = [("t1", "x", False)]
        snap = build_turn_snapshot(
            1, msgs, blocks, outputs, Usage(), "m", 0.0
        )
        msgs.append(Message.user("c"))
        blocks.append(TextBlock(text="d"))
        outputs.append(("t2", "y", True))
        assert len(snap.request_messages) == 1
        assert len(snap.assistant_blocks) == 1
        assert len(snap.tool_outputs) == 1


class TestLoadSession:
    def test_not_found_raises(self, base: Path):
        with pytest.raises(SessionNotFoundError):
            load_session("nope", base)

    async def test_rebuilds_messages_from_jsonl(self, base: Path, tmp_path: Path):
        r = SessionRecorder.open_sync("rb", tmp_path, base_dir=base)
        tool_outputs = [("t1", "file1\nfile2", False)]
        blocks = [
            TextBlock(text="listing"),
            ToolUseBlock(id="t1", name="list_dir", input={"path": "."}),
        ]
        await r.record(_snap(request_messages=[Message.user("列出文件")],
                             assistant_blocks=blocks, tool_outputs=tool_outputs))
        turn2_request = [
            Message.user("列出文件"),
            Message(role=Role.assistant, content=blocks),
            Message(role=Role.tool, content=[
                ToolResultBlock(tool_use_id="t1", content="file1\nfile2")
            ]),
        ]
        await r.record(_snap(iteration=2, request_messages=turn2_request,
                             assistant_blocks=[TextBlock(text="all done")]))
        await r.finalize("ended")

        meta, messages = load_session("rb", base)
        assert meta.id == "rb"
        assert meta.turns == 2
        assert [m.role for m in messages] == [
            Role.user, Role.assistant, Role.tool, Role.assistant,
        ]
        assert messages[1].content[1].name == "list_dir"
        assert messages[2].content[0].tool_use_id == "t1"
        assert messages[2].content[0].is_error is False
        assert messages[-1].content[0].text == "all done"

    async def test_skips_blank_lines(self, base: Path, tmp_path: Path):
        r = SessionRecorder.open_sync("blank", tmp_path, base_dir=base)
        await r.record(_snap())
        day_dir = r._meta_path.parent
        jsonl = day_dir / "blank.jsonl"
        jsonl.write_text(
            jsonl.read_text(encoding="utf-8") + "\n\n",
            encoding="utf-8",
        )
        meta, messages = load_session("blank", base)
        assert meta.turns == 1
        assert len(messages) == 2


class TestListSessions:
    def test_empty_when_no_dir(self, base: Path):
        assert list_sessions(base) == []

    def test_collects_across_days_sorted_desc(self, base: Path, tmp_path: Path):
        for sid, day, started in [
            ("old", "2026-01-01", datetime(2026, 1, 1, tzinfo=UTC)),
            ("mid", "2026-02-01", datetime(2026, 2, 1, tzinfo=UTC)),
            ("new", "2026-03-01", datetime(2026, 3, 1, tzinfo=UTC)),
        ]:
            d = base / day
            d.mkdir(parents=True)
            (d / f"{sid}.meta.json").write_text(
                SessionMeta(
                    id=sid, started_at=started, cwd=str(tmp_path)
                ).model_dump_json(),
                encoding="utf-8",
            )

        metas = list_sessions(base)
        assert [m.id for m in metas] == ["new", "mid", "old"]

    def test_ignores_corrupt_meta(self, base: Path):
        d = base / "2026-01-01"
        d.mkdir(parents=True)
        (d / "bad.meta.json").write_text("{not json", encoding="utf-8")
        assert list_sessions(base) == []


class TestPurgeSessions:
    async def test_purges_old_keeps_new(self, base: Path, tmp_path: Path):
        old_started = datetime.now(UTC) - timedelta(days=40)
        for sid, started in [("ancient", old_started), ("fresh", datetime.now(UTC))]:
            r = SessionRecorder.open_sync(sid, tmp_path, base_dir=base)
            r.meta.started_at = started
            await r.save_meta()
            await r.record(_snap())

        removed = purge_sessions(30, base)
        assert removed == 1
        metas = [m.id for m in list_sessions(base)]
        assert metas == ["fresh"]

    async def test_zero_days_noop(self, base: Path, tmp_path: Path):
        r = SessionRecorder.open_sync("s", tmp_path, base_dir=base)
        await r.record(_snap())
        assert purge_sessions(0, base) == 0

    def test_missing_dir_returns_zero(self, base: Path):
        assert purge_sessions(30, base) == 0


class TestMaybeGenerateTitle:
    async def test_existing_title_skipped(self):
        meta = SessionMeta(
            id="t", started_at=datetime.now(UTC), cwd=".", turns=1, title="已有"
        )
        llm = MockLlm([])
        assert await maybe_generate_title(meta, llm, _snap()) == "已有"
        assert llm.call_count == 0

    async def test_no_user_text_returns_none(self):
        meta = SessionMeta(id="t", started_at=datetime.now(UTC), cwd=".", turns=1)
        snap = _snap(request_messages=[Message(role=Role.assistant, content=[])])
        assert await maybe_generate_title(meta, MockLlm([]), snap) is None

    async def test_generates_title_from_first_user_message(self):
        meta = SessionMeta(id="t", started_at=datetime.now(UTC), cwd=".", turns=1)
        llm = MockLlm([single_text_turn("修复超时问题")])
        snap = _snap(request_messages=[Message.user("帮我修复 rpc client 超时")])
        title = await maybe_generate_title(meta, llm, snap)
        assert title == "修复超时问题"
        assert meta.title == "修复超时问题"

    async def test_llm_failure_returns_none(self):
        meta = SessionMeta(id="t", started_at=datetime.now(UTC), cwd=".", turns=1)

        class BrokenLlm:
            async def complete(self, req):
                msg = "boom"
                raise RuntimeError(msg)

        snap = _snap(request_messages=[Message.user("hello")])
        assert await maybe_generate_title(meta, BrokenLlm(), snap) is None

    async def test_empty_text_not_set(self):
        meta = SessionMeta(id="t", started_at=datetime.now(UTC), cwd=".", turns=1)
        llm = MockLlm([single_text_turn("   ")])
        snap = _snap(request_messages=[Message.user("hello")])
        assert await maybe_generate_title(meta, llm, snap) is None
        assert meta.title is None
