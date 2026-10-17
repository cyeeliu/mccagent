"""AgentLoop 会话接线测试：录制、历史累积与恢复。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from mcc_core import Role, ToolContext, ToolResultBlock, Usage
from mcc_core.event import Notice
from mcc_harness import AgentLoopBuilder
from mcc_llm import CompleteRequest, MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_session import (
    SessionRecorder,
    TurnSnapshot,
    load_session,
)
from mcc_tools import default_registry


class CapturingLlm:
    """包装 MockLlm，记录每次请求供断言。"""

    def __init__(self, turns: list) -> None:
        self._mock = MockLlm(turns)
        self.requests: list[tuple[str, CompleteRequest]] = []

    async def complete(self, req: CompleteRequest):
        self.requests.append(("complete", req))
        return await self._mock.complete(req)

    def stream(self, req: CompleteRequest):
        self.requests.append(("stream", req))
        return self._mock.stream(req)

    @property
    def stream_requests(self) -> list[CompleteRequest]:
        return [r for kind, r in self.requests if kind == "stream"]


def _tool_turn(tool_id: str, tool_name: str, tool_input: dict) -> list:
    return [
        ToolUseStart(id=tool_id, name=tool_name),
        ToolUseInputDelta(partial_json=json.dumps(tool_input)),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="agent-sess")


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


class TestRecorderWiring:
    async def test_end_turn_recorded_and_finalized(self, ctx, base, tmp_path):
        llm = CapturingLlm([
            single_text_turn("你好，有什么可以帮你？"),
            single_text_turn("打招呼测试"),
        ])
        recorder = SessionRecorder.open_sync("r1", tmp_path, base_dir=base)
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        run = await agent.run("hi")

        assert recorder.meta.turns == 1
        assert recorder.meta.cost_usd == pytest.approx(run.cost_usd)
        assert recorder.meta.status == "ended"
        assert recorder.meta.title == "打招呼测试"

        meta, msgs = load_session("r1", base)
        assert meta.turns == 1
        assert msgs[0].content[0].text == "hi"
        assert msgs[-1].content[0].text == "你好，有什么可以帮你？"

    async def test_tool_outputs_recorded(self, ctx, base, tmp_path):
        llm = CapturingLlm([
            _tool_turn("t1", "write_file", {"path": "x.txt", "content": "X"}),
            single_text_turn("written"),
            single_text_turn("写文件测试"),
        ])
        recorder = SessionRecorder.open_sync("r2", tmp_path, base_dir=base)
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        await agent.run("create x.txt")

        _, msgs = load_session("r2", base)
        roles = [m.role for m in msgs]
        assert roles == [Role.user, Role.assistant, Role.tool, Role.assistant]
        tool_results = [
            b for b in msgs[2].content if isinstance(b, ToolResultBlock)
        ]
        assert tool_results[0].tool_use_id == "t1"
        assert tool_results[0].is_error is False
        assert "written" in tool_results[0].content

    async def test_cost_accumulated_across_turns(self, ctx, base, tmp_path):
        usage2 = Usage(input_tokens=2000, output_tokens=200)
        llm = CapturingLlm([
            _tool_turn("c1", "run_bash", {"command": "echo cost"}),
            single_text_turn("多轮测试"),
            single_text_turn("done", usage=usage2),
        ])
        recorder = SessionRecorder.open_sync("r3", tmp_path, base_dir=base)
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        run = await agent.run("go")

        assert recorder.meta.turns == 2
        assert recorder.meta.cost_usd == pytest.approx(run.cost_usd)
        assert recorder.meta.title == "多轮测试"

    async def test_error_finalizes_errored(self, ctx, base, tmp_path):
        class BoomLlm:
            async def complete(self, req):
                msg = "unused"
                raise RuntimeError(msg)

            def stream(self, req):
                msg = "api down"
                raise RuntimeError(msg)

        recorder = SessionRecorder.open_sync("r4", tmp_path, base_dir=base)
        agent = (
            AgentLoopBuilder(BoomLlm(), default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        with pytest.raises(RuntimeError, match="api down"):
            await agent.run("go")
        assert recorder.meta.status == "errored"

    async def test_recorder_failure_not_fatal(self, ctx, tmp_path):
        class BrokenRecorder:
            meta = None

            async def record(self, turn):
                msg = "disk full"
                raise OSError(msg)

            async def finalize(self, status):
                pass

        llm = CapturingLlm([single_text_turn("still ok")])
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .recorder(BrokenRecorder())
            .build()
        )
        evq: asyncio.Queue = asyncio.Queue()
        run = await agent.run_streaming("go", evq)

        assert run.final_text == "still ok"
        events = []
        while not evq.empty():
            events.append(evq.get_nowait())
        assert any(
            isinstance(e, Notice) and "session record failed" in e.text
            for e in events
        )


class TestHistoryAccumulation:
    async def test_history_accumulates_across_runs(self, ctx):
        llm = CapturingLlm([
            single_text_turn("第一轮回答"),
            single_text_turn("标题"),
            single_text_turn("第二轮回答"),
        ])
        agent = AgentLoopBuilder(llm, default_registry(), ctx).build()
        await agent.run("第一问")
        await agent.run("第二问")

        stream_reqs = llm.stream_requests
        assert len(stream_reqs) == 2
        second = stream_reqs[1].messages
        assert second[0].content[0].text == "第一问"
        assert any(m.role == Role.assistant for m in second)
        assert second[-1].content[0].text == "第二问"

    async def test_builder_history_injects_prefix(self, ctx):
        from mcc_core import Message

        llm = CapturingLlm([single_text_turn("ok")])
        history = [
            Message.user("旧问题"),
            Message.assistant("旧回答"),
        ]
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .history(history)
            .build()
        )
        await agent.run("新问题")

        req = llm.stream_requests[0]
        texts = [m.content[0].text for m in req.messages]
        assert texts == ["旧问题", "旧回答", "新问题"]

    async def test_history_not_mutated_by_agent(self, ctx):
        from mcc_core import Message

        llm = CapturingLlm([single_text_turn("ok"), single_text_turn("ok")])
        history = [Message.user("原问题")]
        agent = (
            AgentLoopBuilder(llm, default_registry(), ctx)
            .history(history)
            .build()
        )
        await agent.run("第一次")
        await agent.run("第二次")
        assert len(history) == 1


class TestResume:
    async def test_resume_continues_conversation(self, ctx, base, tmp_path):
        llm1 = CapturingLlm([
            single_text_turn("已记住 uv 偏好"),
            single_text_turn("工具偏好"),
        ])
        recorder = SessionRecorder.open_sync("res1", tmp_path, base_dir=base)
        agent1 = (
            AgentLoopBuilder(llm1, default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        await agent1.run("记住：这个项目用 uv 管理依赖")

        meta, history = load_session("res1", base)
        assert meta.turns == 1

        ctx2 = ToolContext(cwd=tmp_path, session_id="res1")
        llm2 = CapturingLlm([single_text_turn("你之前提到 uv")])
        agent2 = (
            AgentLoopBuilder(llm2, default_registry(), ctx2)
            .history(history)
            .build()
        )
        await agent2.run("我之前说了什么？")

        req = llm2.stream_requests[0]
        assert req.messages[0].content[0].text == "记住：这个项目用 uv 管理依赖"
        assert req.messages[-1].content[0].text == "我之前说了什么？"

    async def test_resume_same_session_appends_jsonl(self, ctx, base, tmp_path):
        llm1 = CapturingLlm([single_text_turn("第一段"), single_text_turn("t")])
        recorder = SessionRecorder.open_sync("res2", tmp_path, base_dir=base)
        agent1 = (
            AgentLoopBuilder(llm1, default_registry(), ctx)
            .recorder(recorder)
            .build()
        )
        await agent1.run("第一问")
        assert recorder.meta.turns == 1

        recorder2 = SessionRecorder.open_sync("res2", tmp_path, base_dir=base)
        assert recorder2.meta.turns == 1

        _, history = load_session("res2", base)
        llm2 = CapturingLlm([single_text_turn("第二段")])
        agent2 = (
            AgentLoopBuilder(llm2, default_registry(), ctx)
            .recorder(recorder2)
            .history(history)
            .build()
        )
        await agent2.run("第二问")

        _, history = load_session("res2", base)
        user_texts = [
            m.content[0].text for m in history if m.role == Role.user
        ]
        assert "第一问" in user_texts and "第二问" in user_texts

    def test_turnsnapshot_roundtrip_via_agent(
        self, ctx, base, tmp_path
    ):
        """完整回放：JSONL 每行可解析为 TurnSnapshot。"""
        import asyncio as aio

        async def run():
            llm = CapturingLlm([
                single_text_turn("done"),
                single_text_turn("回放测试"),
            ])
            recorder = SessionRecorder.open_sync("rt", tmp_path, base_dir=base)
            agent = (
                AgentLoopBuilder(llm, default_registry(), ctx)
                .recorder(recorder)
                .build()
            )
            await agent.run("play")

        aio.run(run())
        day_dir = base / list(base.iterdir())[0].name
        lines = (day_dir / "rt.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        turn = TurnSnapshot.model_validate_json(lines[0])
        assert turn.request_messages[0].content[0].text == "play"
        assert turn.assistant_blocks[0].type == "text"
