"""端到端测试：/resume 在 TUI 模式下的事件流。

验证两个用户报告的问题：
1. session 列表是否标识当前会话
2. 选中 session 后 replay 事件是否正确生成
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core import (
    Notice,
    Role,
    SelectionRequest,
    SessionChanged,
    TextBlock,
    TextDelta,
    TurnEnd,
    UserEcho,
)
from mcc_session import SessionRecorder


class _MockAgent:
    """最小 Agent mock：仅支持 history 赋值。"""

    def __init__(self):
        self.history = []
        self.model = "test-model"
        self.permission_responses = None

    async def run_streaming(self, prompt, queue):
        yield  # make it an async generator


async def _create_session_with_history(
    work_dir: Path, sid: str, user_text: str, assistant_text: str
) -> str:
    """创建一个有真实历史的 session，返回 session id。"""
    from datetime import UTC, datetime

    from mcc_core import Message
    from mcc_session import TurnSnapshot

    recorder = SessionRecorder.open_sync(sid, work_dir)
    turn = TurnSnapshot(
        ts=datetime.now(UTC),
        iteration=0,
        request_messages=[Message(role=Role.user, content=[TextBlock(text=user_text)])],
        assistant_blocks=[TextBlock(text=assistant_text)],
        tool_outputs=[],
        usage={"input_tokens": 10, "output_tokens": 5},
        cost_usd=0.01,
    )
    await recorder.record(turn)
    return sid


@pytest.mark.asyncio
async def test_resume_full_flow_through_tui_mode(tmp_path: Path):
    """/resume 完整流程：列出 → 选择 → replay 事件正确生成。"""
    import mcc_cli
    import mcc_tui
    from mcc_cli.events import AgentEventSource
    from mcc_cli.slash import build_default_registry

    # 1. 创建一个有历史的 session
    sid = "abc12345"
    await _create_session_with_history(
        tmp_path, sid, "hello world", "Hi! How can I help?"
    )

    # 2. 创建当前会话（模拟用户刚启动 mcc）
    current_sid = "cur99999"
    SessionRecorder.open_sync(current_sid, tmp_path)

    # 3. 构建 AgentEventSource
    agent = _MockAgent()
    selection_responses: asyncio.Queue = asyncio.Queue()
    source = AgentEventSource(
        agent,
        slash_registry=build_default_registry(),
        session_short_id=current_sid,
        selection_responses=selection_responses,
    )

    captured: list = []

    # 4. Patch run_tui：模拟用户输入 /resume → 选择 session → 收集事件
    original_run_tui = mcc_tui.run_tui

    async def fake_run_tui(handles, console=None):
        # 4a. 用户输入 /resume
        handles.input_tx.put_nowait("/resume")

        # 4b. 等待 SelectionRequest 事件出现
        selection_req = None
        for _ in range(200):
            await asyncio.sleep(0.01)
            while not handles.events.empty():
                ev = handles.events.get_nowait()
                captured.append(ev)
                if isinstance(ev, SelectionRequest):
                    selection_req = ev
            if selection_req is not None:
                break

        assert selection_req is not None, "未收到 SelectionRequest"

        # 4c. 验证当前会话被标记
        labels = [opt.label for opt in selection_req.options]
        current_marked = any("●" in lbl and current_sid in lbl for lbl in labels)
        assert current_marked, (
            f"当前会话 {current_sid} 未被 ● 标记。labels:\n"
            + "\n".join(labels)
        )

        # 4c-2. 验证当前会话排在第一位
        assert current_sid in labels[0], (
            "当前会话未排在第一位。labels:\n" + "\n".join(labels)
        )

        # 4d. 用户选择有历史的 session
        target_option = None
        for opt in selection_req.options:
            if opt.value == sid:
                target_option = opt
                break
        assert target_option is not None, f"目标 session {sid} 不在列表中"

        from mcc_core import SelectionDecision

        handles.selection_tx.put_nowait(
            SelectionDecision(request_id=selection_req.id, value=sid)
        )

        # 4e. 收集后续所有事件直到 TurnEnd
        turn_end_count = 0
        for _ in range(500):
            await asyncio.sleep(0.01)
            while not handles.events.empty():
                ev = handles.events.get_nowait()
                captured.append(ev)
                if isinstance(ev, TurnEnd):
                    turn_end_count += 1
            if turn_end_count >= 1:
                # 多等一轮确保所有事件都到达
                await asyncio.sleep(0.05)
                while not handles.events.empty():
                    captured.append(handles.events.get_nowait())
                break

    mcc_tui.run_tui = fake_run_tui
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        mcc_tui.run_tui = original_run_tui

    # 5. 验证事件流
    # 5a. SelectionRequest 存在
    sel_reqs = [e for e in captured if isinstance(e, SelectionRequest)]
    assert len(sel_reqs) == 1

    # 5b. SessionChanged 存在
    session_changes = [e for e in captured if isinstance(e, SessionChanged)]
    assert len(session_changes) >= 1
    assert session_changes[0].session_id == sid

    # 5c. Notice 存在（"已恢复..."）
    notices = [e for e in captured if isinstance(e, Notice)]
    assert any("已恢复" in n.text for n in notices)

    # 5d. UserEcho 存在（历史中的用户消息）
    user_echos = [e for e in captured if isinstance(e, UserEcho)]
    assert len(user_echos) >= 1
    assert any("hello world" in ue.text for ue in user_echos)

    # 5e. TextDelta 存在（历史中的助手消息）
    text_deltas = [e for e in captured if isinstance(e, TextDelta)]
    assert len(text_deltas) >= 1
    assert any("Hi! How can I help?" in td.text for td in text_deltas)

    # 5f. TurnEnd 存在
    turn_ends = [e for e in captured if isinstance(e, TurnEnd)]
    assert len(turn_ends) >= 1


@pytest.mark.asyncio
async def test_resume_marks_current_session(tmp_path: Path):
    """验证 /resume 列表中当前会话被 ● 标记。"""
    from mcc_cli.events import AgentEventSource
    from mcc_cli.slash import build_default_registry

    # 创建几个 session
    sid1 = "sess0001"
    sid2 = "sess0002"
    current_sid = "curr0003"
    await _create_session_with_history(tmp_path, sid1, "msg1", "reply1")
    await _create_session_with_history(tmp_path, sid2, "msg2", "reply2")
    SessionRecorder.open_sync(current_sid, tmp_path)

    agent = _MockAgent()
    selection_responses: asyncio.Queue = asyncio.Queue()
    source = AgentEventSource(
        agent,
        slash_registry=build_default_registry(),
        session_short_id=current_sid,
        selection_responses=selection_responses,
    )

    events = []
    async for ev in source.run("/resume"):
        events.append(ev)
        if isinstance(ev, SelectionRequest):
            break

    sel_req = next(e for e in events if isinstance(e, SelectionRequest))
    labels = [opt.label for opt in sel_req.options]

    # 当前会话应被标记
    current_labels = [label for label in labels if current_sid in label]
    assert len(current_labels) == 1
    assert "●" in current_labels[0], f"当前会话未被 ● 标记: {current_labels[0]}"
    assert "(当前)" in current_labels[0], f"当前会话未显示 (当前) 后缀: {current_labels[0]}"

    # 非当前会话不应被标记
    other_labels = [label for label in labels if sid1 in label]
    assert len(other_labels) == 1
    assert "●" not in other_labels[0], f"非当前会话被误标记: {other_labels[0]}"

    # 当前会话应排在第一位
    assert current_sid in labels[0], f"当前会话未排在第一位: {labels[0]}"


@pytest.mark.asyncio
async def test_tui_mode_uses_source_session_short_id(tmp_path: Path):
    """_run_tui_mode 应复用 source.session_short_id 而非生成新随机 ID。"""
    import mcc_cli
    import mcc_tui
    from mcc_cli.events import AgentEventSource
    from mcc_cli.slash import build_default_registry

    agent = _MockAgent()
    source_sid = "src12345"
    source = AgentEventSource(
        agent,
        slash_registry=build_default_registry(),
        session_short_id=source_sid,
    )

    captured_sid: dict = {}

    original_run_tui = mcc_tui.run_tui

    async def fake_run_tui(handles, console=None):
        captured_sid["handles"] = handles.session_short_id

    mcc_tui.run_tui = fake_run_tui
    try:
        await mcc_cli._run_tui_mode(source, work_dir=tmp_path)
    finally:
        mcc_tui.run_tui = original_run_tui

    assert captured_sid["handles"] == source_sid, (
        f"TuiHandles.session_short_id={captured_sid.get('handles')} "
        f"!= source.session_short_id={source_sid}"
    )
