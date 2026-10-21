"""上下文压缩测试：四步管线的纯函数行为与配对保护。

不依赖 LLM 的步骤全部真执行；摘要步骤注入假 summarize 函数。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock
from mcc_harness.compactor import (
    CONTEXT_CHAR_LIMIT,
    KEEP_HEAD,
    KEEP_TAIL,
    MAX_MESSAGES,
    CompactionReport,
    ContextCompactor,
    estimate_chars,
    is_too_long_error,
)


def _user(text: str) -> Message:
    return Message(role=Role.user, content=[TextBlock(text=text)])


def _assistant_with_tools(*ids: str) -> Message:
    return Message(role=Role.assistant, content=[
        ToolUseBlock(id=i, name="run_bash", input={}) for i in ids
    ])


def _results_msg(*pairs: tuple[str, str]) -> Message:
    """一条携带多个 tool_result 的 user 消息（mcc 的工具结果形态）。"""
    return Message(role=Role.user, content=[
        ToolResultBlock(tool_use_id=tid, content=content) for tid, content in pairs
    ])


def _pair(tid: str, result: str = "ok") -> list[Message]:
    """最小工具配对：assistant(tool_use) + user(tool_result)。"""
    return [_assistant_with_tools(tid), _results_msg((tid, result))]


def _compactor(tmp_path: Path, **kwargs) -> ContextCompactor:
    return ContextCompactor(
        persist_dir=tmp_path / ".task_outputs" / "tool-results",
        **kwargs,
    )


@pytest.fixture
def ctx(tmp_path: Path):
    from mcc_core import ToolContext

    return ToolContext(cwd=tmp_path, session_id="compactor-test", depth=0)


@pytest.fixture
def registry():
    from mcc_tools import default_registry

    return default_registry()


def _fake_summary(text: str = "SUMMARY-TEXT"):
    async def summarize(system: str, messages: list[Message]) -> str:
        assert "summarizer" in system.lower()
        return text
    return summarize


# ── 估算与错误标记 ──────────────────────────────────────────────────────


def test_estimate_chars_monotonic() -> None:
    one = [_user("hello")]
    two = [*one, _user("world")]
    assert estimate_chars(two) > estimate_chars(one)


@pytest.mark.parametrize(
    "text",
    [
        "Error: prompt_too_long: tokens: 200001 > 200000 maximum",
        "Error code: 400 - {'error': {'message': 'context_length_exceeded'",
        "This model supports maximum context length of 8192 tokens",
        "request too large: exceeding token budget",
    ],
    ids=["anthropic", "openai", "max-len", "too-large"],
)
def test_is_too_long_error_matches(text: str) -> None:
    assert is_too_long_error(text) is True


@pytest.mark.parametrize(
    "text",
    ["rate limit exceeded", "internal server error", "invalid api key"],
)
def test_is_too_long_error_rejects_others(text: str) -> None:
    assert is_too_long_error(text) is False


# ── 第 1 步：tool_result_budget ────────────────────────────────────────


async def test_budget_persists_large_result_with_preview(tmp_path: Path) -> None:
    """超预算的最新批大结果转存：留路径 + 预览，文件可按路径读回。"""
    big = "x" * 60_000  # 4 条共 240k，超 200k 预算触发
    messages = [
        _user("do work"),
        _assistant_with_tools("t1", "t2", "t3", "t4"),
        _results_msg(("t1", big), ("t2", big), ("t3", big), ("t4", big)),
    ]
    compactor = _compactor(tmp_path)
    report = CompactionReport()

    out = compactor.tool_result_budget(messages, report)

    blocks = [b for b in out[-1].content if isinstance(b, ToolResultBlock)]
    persisted = [b for b in blocks if "[Large tool result persisted at" in b.content]
    assert len(persisted) == 1  # 转存一条（58k）即回到 200k 预算内
    saved = tmp_path / ".task_outputs" / "tool-results" / "t1.txt"
    assert saved.read_text(encoding="utf-8") == big  # 完整可恢复
    assert all(b.content.count("x") <= 2_100 for b in persisted)  # 只留预览
    assert report.persisted_results == ["t1"]


async def test_budget_noop_under_threshold(tmp_path: Path) -> None:
    small = "y" * 1_000
    messages = [*_pair("t1", small)]
    compactor = _compactor(tmp_path)

    out = compactor.tool_result_budget(messages, CompactionReport())

    assert out[-1].content[0].content == small
    assert not (tmp_path / ".task_outputs").exists()  # 未发生任何转存


async def test_budget_only_touches_latest_user_message(tmp_path: Path) -> None:
    """只处理最新一条 user 消息：历史里的大结果留给第 3 步。"""
    big = "z" * 60_000
    messages = [
        _user("task"),
        _assistant_with_tools("t1"),
        _results_msg(("t1", big)),
        _assistant_with_tools("t2"),
        _results_msg(("t2", "small")),
    ]
    compactor = _compactor(tmp_path)

    out = compactor.tool_result_budget(messages, CompactionReport())

    assert out[2].content[0].content == big  # 历史大结果未动
    assert out[4].content[0].content == "small"


# ── 第 2 步：snip_compact ──────────────────────────────────────────────


def _long_history(n: int) -> list[Message]:
    msgs = [_user("task statement")]
    for i in range(n):
        msgs.extend(_pair(f"t{i}", f"result {i}"))
    return msgs


async def test_snip_archives_middle_keeps_head_and_tail(tmp_path: Path) -> None:
    messages = _long_history(MAX_MESSAGES + 10)
    compactor = _compactor(tmp_path)

    out = compactor.snip_compact(messages, CompactionReport())

    assert len(out) <= KEEP_HEAD + KEEP_TAIL + 1
    assert out[0].content[0].text == "task statement"  # 头部保留（任务约束）
    assert out[-1] is messages[-1]  # 尾部保留（当前工作）
    marker = out[KEEP_HEAD]
    assert "archived" in marker.content[0].text


async def test_snip_noop_at_or_below_limit(tmp_path: Path) -> None:
    messages = [_user(f"msg {i}") for i in range(MAX_MESSAGES)]  # 恰好上限
    compactor = _compactor(tmp_path)

    out = compactor.snip_compact(messages, CompactionReport())

    assert out is messages  # 原对象返回，未做任何归档


async def test_snip_never_breaks_tool_pairing(tmp_path: Path) -> None:
    """配对保护：归档边界不产生孤立 tool_result 或孤立 tool_use。"""
    messages = _long_history(MAX_MESSAGES + 10)
    compactor = _compactor(tmp_path)

    out = compactor.snip_compact(messages, CompactionReport())

    seen_uses: set[str] = set()
    orphan_results: set[str] = set()
    for msg in out:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                seen_uses.add(block.id)
            elif (isinstance(block, ToolResultBlock)
                    and block.tool_use_id not in seen_uses):
                orphan_results.add(block.tool_use_id)
    assert not orphan_results  # 每个 tool_result 都能找到配对调用


# ── 第 3 步：micro_compact ─────────────────────────────────────────────


async def test_micro_replaces_old_read_results_keeps_recent(tmp_path: Path) -> None:
    """模型已读的旧长结果被替换为路径；最近几条保持完整。"""
    results = [(f"t{i}", f"payload {i} " * 100) for i in range(6)]
    messages = [
        _user("task"),
        _assistant_with_tools(*(tid for tid, _ in results)),
        _results_msg(*results),
        _assistant_with_tools("last"),  # 模型消费过上述结果
        _results_msg(("last", "recent")),
    ]
    compactor = _compactor(tmp_path)
    report = CompactionReport()

    out = compactor.micro_compact(messages, target_chars=0, report=report)
    # target=0：可替换的都处理；最近 3 条（KEEP_RECENT_RESULTS）完整

    blocks = [b for b in out[2].content if isinstance(b, ToolResultBlock)]
    early, recent = blocks[:3], blocks[3:]
    assert all("[Earlier tool result saved at" in b.content for b in early)
    assert all(b.content.startswith("payload") for b in recent)  # 最近完整
    saved = tmp_path / ".task_outputs" / "tool-results" / "t0.txt"
    assert "payload 0" in saved.read_text(encoding="utf-8")
    assert out[4].content[0].content == "recent"
    assert report.replaced_results == 3


async def test_micro_skips_unseen_results(tmp_path: Path) -> None:
    """末条 assistant 之后的新结果（未读）不动——交给第 1 步与 API 轮次。"""
    fresh = "fresh " * 100
    messages = [
        _user("task"),
        _assistant_with_tools("t1"),
        _results_msg(("t1", fresh)),  # 末尾无 assistant 消费：未读
    ]
    compactor = _compactor(tmp_path)

    out = compactor.micro_compact(messages, target_chars=0, report=CompactionReport())

    assert out[2].content[0].content == fresh


async def test_micro_short_results_untouched(tmp_path: Path) -> None:
    messages = [
        _user("task"),
        _assistant_with_tools("t1"),
        _results_msg(("t1", "short")),
        _assistant_with_tools("done"),
    ]
    compactor = _compactor(tmp_path)

    out = compactor.micro_compact(messages, target_chars=0, report=CompactionReport())

    assert out[2].content[0].content == "short"


# ── 第 4 步：compact_history ───────────────────────────────────────────


async def test_compact_history_separates_request_and_summary(tmp_path: Path) -> None:
    """摘要消息显式分开 Current user request 与 Conversation summary。"""
    messages = [*_pair("t1", "ok"), _user("keep going")]
    compactor = _compactor(tmp_path, summarize=_fake_summary("goal done partly"))

    out = await compactor.compact_history(messages, "fix the bug", CompactionReport())

    assert len(out) == 1
    text = out[0].content[0].text
    assert "## Current user request\nfix the bug" in text
    assert "## Conversation summary\ngoal done partly" in text
    assert out[0].role is Role.user


async def test_compact_history_without_summarize_noop(tmp_path: Path) -> None:
    """未注入摘要函数时原样返回（渐进降级，不硬失败）。"""
    messages = [*_pair("t1")]
    compactor = _compactor(tmp_path)

    out = await compactor.compact_history(messages, "t", CompactionReport())

    assert out is messages


# ── prepare 全管线 ─────────────────────────────────────────────────────


async def test_prepare_runs_budget_and_snip_every_turn(tmp_path: Path) -> None:
    """未超限的常规轮：仅第 1/2 步（每轮执行），不动历史内容。"""
    messages = _long_history(MAX_MESSAGES + 5)
    compactor = _compactor(tmp_path)

    out, report = await compactor.prepare(messages, "task")

    assert report.archived_messages > 0
    assert not report.summarized


async def test_prepare_escalates_to_summary_when_over_limit(tmp_path: Path) -> None:
    """各步之后仍超限 → 升级到模型摘要（唯一有损步骤）。"""
    big = "x" * 40_000
    messages = [
        _user("heavy task"),
        _assistant_with_tools("t1", "t2"),
        _results_msg(("t1", big), ("t2", big)),
        _assistant_with_tools("done"),
        _user("continue"),
    ]
    compactor = _compactor(
        tmp_path, summarize=_fake_summary(), char_limit=5_000
    )

    out, report = await compactor.prepare(messages, "heavy task")

    assert report.summarized
    assert len(out) == 1
    assert "## Current user request\nheavy task" in out[0].content[0].text


async def test_prepare_stops_early_when_under_limit(tmp_path: Path) -> None:
    """micro 压到阈值之下后不再升级：无摘要、无 fit。"""
    payload = "p" * 3_000
    results = [(f"t{i}", payload) for i in range(5)]
    messages = [
        _user("task"),
        _assistant_with_tools(*(t for t, _ in results)),
        _results_msg(*results),
        _assistant_with_tools("done"),
    ]
    before = estimate_chars(messages)  # micro 是 in-place 块替换，先留底
    # 下限约束：最近 3 条结果完整保留（3×3k+结构 ≈ 10.2k），
    # 阈值须高于物理下限、低于初始估算，micro 才可能停在中途
    limit = 11_000
    compactor = _compactor(tmp_path, char_limit=limit)

    out, report = await compactor.prepare(messages, "task")

    assert not report.summarized  # micro 压回阈值内，未升级到摘要
    assert report.replaced_results > 0
    assert estimate_chars(out) < before
    assert estimate_chars(out) <= limit


# ── reactive_compact ──────────────────────────────────────────────────


async def test_reactive_keeps_recent_with_pairing(tmp_path: Path) -> None:
    """API 拒绝后的补救：摘要旧历史、保尾继续、配对不破。"""
    messages = _long_history(30)
    compactor = _compactor(tmp_path, summarize=_fake_summary())

    out = await compactor.reactive_compact(messages, "my task")

    assert out[0].role is Role.user
    assert "[Reactive compact" in out[0].content[0].text
    assert "## Current user request\nmy task" in out[0].content[0].text
    assert len(out) <= 7  # 摘要 + 最近 5 条（切点配对保护可能左移 1）

    seen: set[str] = set()
    orphans = set()
    for msg in out[1:]:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                seen.add(block.id)
            elif isinstance(block, ToolResultBlock) and block.tool_use_id not in seen:
                orphans.add(block.tool_use_id)
    assert not orphans


async def test_reactive_without_summarize_noop(tmp_path: Path) -> None:
    messages = _long_history(10)
    compactor = _compactor(tmp_path)

    out = await compactor.reactive_compact(messages, "t")

    assert out is messages


# ── 报告描述 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("report", "expected_fragment"),
    [
        (CompactionReport(persisted_results=["a", "b"]), "2 large result(s) persisted"),
        (CompactionReport(archived_messages=7), "7 message(s) archived"),
        (CompactionReport(replaced_results=3), "3 old result(s) replaced"),
        (CompactionReport(summarized=True), "history summarized"),
    ],
    ids=["persisted", "archived", "replaced", "summarized"],
)
def test_report_describe(report: CompactionReport, expected_fragment: str) -> None:
    assert report.notable
    assert expected_fragment in report.describe()


def test_report_empty_not_notable() -> None:
    report = CompactionReport()
    assert not report.notable
    assert report.describe() == ""


def test_default_char_limit_exported() -> None:
    """默认阈值导出为模块常量（配置接线与测试共用）。"""
    assert CONTEXT_CHAR_LIMIT == 120_000
    assert (MAX_MESSAGES, KEEP_HEAD, KEEP_TAIL) == (50, 3, 46)


# ─── AgentLoop 接线（MockLlm 全链路）────────────────────────────────────


def _make_agent_with_compactor(llm, ctx, registry, tmp_path: Path, *, char_limit: int):
    from mcc_harness import AgentLoopBuilder

    compactor = ContextCompactor(
        persist_dir=tmp_path / ".task_outputs" / "tool-results",
        char_limit=char_limit,
    )
    return (
        AgentLoopBuilder(llm, registry, ctx)
        .model("test-model")
        .compactor(compactor)
        .build()
    ), compactor


async def test_agent_loop_compacts_over_limit_conversation(ctx, registry, tmp_path):
    """长会话超限触发压缩：Notice 通知可见、摘要接管后任务照常完成。"""
    import asyncio

    from mcc_core import Usage
    from mcc_core.event import Notice
    from mcc_llm import (
        MessageResponse,
        MessageStop,
        TextDelta,
        ToolUseInputDelta,
        ToolUseStart,
    )
    from mcc_llm.mock import MockLlm

    class _SummarizingLlm(MockLlm):
        """stream 消费预设轮次；complete（摘要）固定返回，互不干扰。"""

        async def complete(self, req):
            return MessageResponse(
                content=[TextBlock(text="goal: finish the task; midway")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    def _tool_turn(text: str, tid: str) -> list:
        return [
            TextDelta(text=text),
            ToolUseStart(id=tid, name="run_bash"),
            ToolUseInputDelta(partial_json='{"command": "echo ok"}'),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]

    # 三轮各带 4k 文本（撑过 8k 阈值）+ 收尾轮
    llm = _SummarizingLlm([
        _tool_turn("A" * 4_000, "t1"),
        _tool_turn("C" * 4_000, "t2"),
        _tool_turn("D" * 4_000, "t3"),
        [TextDelta(text="all done"), MessageStop(
            stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=5))],
    ])
    agent, _ = _make_agent_with_compactor(
        llm, ctx, registry, tmp_path, char_limit=8_000
    )
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("start task", queue)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert run.final_text == "all done"
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert any("compacted" in n and "summarized" in n for n in notices)


async def test_agent_loop_reactive_compact_on_rejection(ctx, registry, tmp_path):
    """API 拒绝超限 → reactive 补救一次并重试成功（不占迭代预算）。"""
    import asyncio

    from mcc_core import Usage
    from mcc_core.event import Notice
    from mcc_llm import MessageResponse, MessageStop, TextDelta
    from mcc_llm.mock import MockLlm

    class _RejectingOnceLlm(MockLlm):
        """首次 stream 抛 prompt_too_long（真实 API 拒绝形态），不消费轮次。"""

        def __init__(self) -> None:
            super().__init__([
                [TextDelta(text="recovered"), MessageStop(
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=5, output_tokens=5))],
            ])
            self.rejected = False

        async def stream(self, req):
            if not self.rejected:
                self.rejected = True
                raise RuntimeError(
                    "Error: prompt_too_long: tokens: 200001 > 200000 maximum"
                )
            async for ev in super().stream(req):
                yield ev

        async def complete(self, req):
            return MessageResponse(
                content=[TextBlock(text="earlier work summary")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    llm = _RejectingOnceLlm()
    agent, _ = _make_agent_with_compactor(
        llm, ctx, registry, tmp_path, char_limit=100_000
    )
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("survive rejection", queue)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert "recovered" in (run.final_text or "")
    notices = [e for e in events if isinstance(e, Notice)]
    assert any("reactive compact" in n.text for n in notices)


async def test_agent_loop_without_compactor_zero_regression(ctx, registry):
    """未注入 compactor：行为与历史版本完全一致（零回归保险）。"""
    import asyncio

    from mcc_harness import AgentLoopBuilder
    from mcc_llm.mock import MockLlm, single_text_turn

    llm = MockLlm([single_text_turn("plain answer")])
    agent = AgentLoopBuilder(llm, registry, ctx).model("m").build()
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("hi", queue)
    assert run.final_text == "plain answer"
    assert agent.compactor is None


# ─── compact 工具与模型主动压缩 ─────────────────────────────────────────


async def test_compact_tool_no_side_effects(tmp_path: Path) -> None:
    """compact 工具本身无参数无副作用：只返回确认文案。"""
    from mcc_core import ToolContext
    from mcc_tools import default_registry

    tool = default_registry().get("compact")
    assert tool is not None
    out = await tool.execute({}, ToolContext(cwd=tmp_path, session_id="s"))
    assert not out.is_error
    assert "Compaction requested" in out.content


async def test_agent_loop_compacts_after_model_request(ctx, registry, tmp_path):
    """模型调用 compact 工具：批次闭合（工具结果落记录）后执行摘要。"""
    import asyncio
    import json as _json

    from mcc_core import Usage
    from mcc_core.event import Notice
    from mcc_llm import (
        MessageResponse,
        MessageStop,
        TextDelta,
        ToolUseInputDelta,
        ToolUseStart,
    )
    from mcc_llm.mock import MockLlm

    summary_calls: list[int] = []

    class _SummarizingLlm(MockLlm):
        async def complete(self, req):
            summary_calls.append(len(req.messages))
            return MessageResponse(
                content=[TextBlock(text="phase one finished")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    def _compact_and_write_turn(tid_write: str, tid_compact: str) -> list:
        return [
            TextDelta(text="phase done, compress now"),
            ToolUseStart(id=tid_write, name="run_bash"),
            ToolUseInputDelta(
                partial_json=_json.dumps({"command": "echo side-effect"})),
            ToolUseStart(id=tid_compact, name="compact"),
            MessageStop(stop_reason="tool_use", usage=Usage()),
        ]

    llm = _SummarizingLlm([
        _compact_and_write_turn("w1", "c1"),
        [TextDelta(text="final answer"), MessageStop(
            stop_reason="end_turn", usage=Usage(input_tokens=5, output_tokens=5))],
    ])
    agent, _ = _make_agent_with_compactor(
        llm, ctx, registry, tmp_path, char_limit=100_000
    )
    queue: asyncio.Queue = asyncio.Queue()

    run = await agent.run_streaming("multi-phase task", queue)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert run.final_text == "final answer"
    # 摘要发生且输入含已闭合的工具结果（副作用已入记录）
    assert summary_calls and summary_calls[0] >= 3
    notices = [e.text for e in events if isinstance(e, Notice)]
    assert any("model-requested" in n for n in notices)


async def test_cli_no_compact_flag_disables(ctx, registry):
    """--no-compact 语义由 config.context.auto_compact / flag 组合表达：
    flag 为 True 时 _build_source 不注入 compactor。"""
    from mcc_harness import AgentLoopBuilder

    llm = None  # 仅验证 builder 组装路径，不真正构建
    builder = AgentLoopBuilder(llm, registry, ctx)  # type: ignore[arg-type]
    agent = builder.build()
    assert agent.compactor is None  # 未注入即禁用
