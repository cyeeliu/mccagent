"""ResultCollector 结果收集器测试。

覆盖：
- 类型定义与校验（ArtifactRef, ExecutionMetrics, AgentResult）
- 结果汇总（ResultSummary 聚合属性）
- 收集器记录/查询/更新/清除
- 产物引用检索
- 序列化与反序列化
- Protocol 异步接口
- 端到端多 Agent 结果汇总
"""

from __future__ import annotations

import pytest
from mcc_harness.results import (
    AgentResult,
    ArtifactKind,
    ArtifactRef,
    ExecutionMetrics,
    ResultAlreadyExistsError,
    ResultCollectorImpl,
    ResultError,
    ResultNotFoundError,
    ResultStatus,
    ResultSummary,
)

# -- 辅助函数 --


def make_result(
    agent_id: str = "a1",
    task_id: str = "t1",
    status: ResultStatus = ResultStatus.SUCCESS,
    *,
    run_id: str | None = "run1",
    summary: str = "done",
    error: str | None = None,
    cost: float = 0.01,
    input_tokens: int = 100,
    output_tokens: int = 50,
    artifacts: tuple[ArtifactRef, ...] = (),
) -> AgentResult:
    metrics = ExecutionMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        started_at=1000.0,
        finished_at=1005.0,
    )
    return AgentResult(
        agent_id=agent_id,
        task_id=task_id,
        status=status,
        summary=summary,
        error=error,
        metrics=metrics,
        artifacts=artifacts,
        team_run_id=run_id,
    )


def make_artifact(
    path: str = "output.txt",
    kind: ArtifactKind = ArtifactKind.FILE,
) -> ArtifactRef:
    return ArtifactRef(path=path, kind=kind, size_bytes=1024)


# -- ResultStatus --


class TestResultStatus:
    """结果状态枚举。"""

    def test_success_is_not_failure(self):
        assert ResultStatus.SUCCESS == "success"

    def test_failure_statuses(self):
        failures = {
            ResultStatus.FAILED,
            ResultStatus.TIMEOUT,
            ResultStatus.CANCELLED,
            ResultStatus.BUDGET_EXCEEDED,
        }
        for s in failures:
            assert s != ResultStatus.SUCCESS
            assert s != ResultStatus.PARTIAL

    @pytest.mark.parametrize("status,expect_failure", [
        (ResultStatus.SUCCESS, False),
        (ResultStatus.PARTIAL, False),
        (ResultStatus.FAILED, True),
        (ResultStatus.TIMEOUT, True),
        (ResultStatus.CANCELLED, True),
        (ResultStatus.BUDGET_EXCEEDED, True),
    ])
    def test_is_failure_property(self, status, expect_failure):
        result = make_result(status=status)
        assert result.is_failure == expect_failure


# -- ArtifactRef --


class TestArtifactRef:
    """产物引用。"""

    def test_default_creation(self):
        ref = ArtifactRef(path="result.txt")
        assert ref.path == "result.txt"
        assert ref.kind == ArtifactKind.FILE
        assert ref.size_bytes is None
        assert ref.checksum is None

    def test_full_creation(self):
        ref = ArtifactRef(
            path="diff.patch",
            kind=ArtifactKind.DIFF,
            description="generated diff",
            size_bytes=2048,
            checksum="abc123",
        )
        assert ref.kind == ArtifactKind.DIFF
        assert ref.size_bytes == 2048

    @pytest.mark.parametrize("kind", list(ArtifactKind))
    def test_all_kinds(self, kind):
        ref = ArtifactRef(path="f", kind=kind)
        assert ref.kind == kind

    def test_empty_path_raises(self):
        with pytest.raises(ResultError, match="path must not be empty"):
            ArtifactRef(path="")

    def test_negative_size_raises(self):
        with pytest.raises(ResultError, match="size_bytes must be >= 0"):
            ArtifactRef(path="f", size_bytes=-1)


# -- ExecutionMetrics --


class TestExecutionMetrics:
    """执行度量。"""

    def test_defaults(self):
        m = ExecutionMetrics()
        assert m.input_tokens == 0
        assert m.output_tokens == 0
        assert m.cost_usd == 0.0
        assert m.total_tokens == 0
        assert m.duration_secs is None

    def test_total_tokens(self):
        m = ExecutionMetrics(input_tokens=200, output_tokens=100)
        assert m.total_tokens == 300

    def test_duration_secs(self):
        m = ExecutionMetrics(started_at=100.0, finished_at=105.5)
        assert m.duration_secs == pytest.approx(5.5)

    def test_duration_none_without_finished(self):
        m = ExecutionMetrics(started_at=100.0)
        assert m.duration_secs is None

    @pytest.mark.parametrize("field", [
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    ])
    def test_negative_tokens_raise(self, field):
        with pytest.raises(ResultError, match=f"{field} must be >= 0"):
            ExecutionMetrics(**{field: -1})

    def test_negative_cost_raises(self):
        with pytest.raises(ResultError, match="cost_usd must be >= 0"):
            ExecutionMetrics(cost_usd=-0.01)

    def test_cache_tokens(self):
        m = ExecutionMetrics(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=80,
            cache_write_tokens=20,
        )
        assert m.cache_read_tokens == 80
        assert m.cache_write_tokens == 20
        assert m.total_tokens == 150


# -- AgentResult --


class TestAgentResult:
    """单个 Agent 执行结果。"""

    def test_success_result(self):
        r = make_result(status=ResultStatus.SUCCESS)
        assert r.is_success
        assert not r.is_failure
        assert not r.is_partial

    def test_partial_result(self):
        r = make_result(status=ResultStatus.PARTIAL)
        assert r.is_partial
        assert not r.is_success
        assert not r.is_failure

    def test_failed_result(self):
        r = make_result(status=ResultStatus.FAILED, error="boom")
        assert r.is_failure
        assert not r.is_success
        assert r.error == "boom"

    def test_with_artifacts(self):
        artifacts = (make_artifact("a.txt"), make_artifact("b.log", ArtifactKind.LOG))
        r = make_result(artifacts=artifacts)
        assert r.has_artifacts
        assert len(r.artifacts) == 2

    def test_no_artifacts(self):
        r = make_result()
        assert not r.has_artifacts

    def test_trace_and_run_association(self):
        r = AgentResult(
            agent_id="a1",
            task_id="t1",
            status=ResultStatus.SUCCESS,
            team_run_id="run-42",
            trace_id="trace-99",
        )
        assert r.team_run_id == "run-42"
        assert r.trace_id == "trace-99"

    def test_meta(self):
        r = AgentResult(
            agent_id="a1",
            task_id="t1",
            status=ResultStatus.SUCCESS,
            meta={"checks": 5, "passed": 5},
        )
        assert r.meta["checks"] == 5


# -- ResultSummary --


class TestResultSummary:
    """结果汇总。"""

    def test_empty_summary(self):
        s = ResultSummary(run_id="r1")
        assert s.total == 0
        assert s.succeeded == 0
        assert s.failed == 0
        assert s.success_rate == 0.0
        assert not s.all_succeeded
        assert not s.has_failures

    def test_all_succeeded(self):
        results = (
            make_result("a1", "t1", run_id="r1"),
            make_result("a2", "t2", run_id="r1"),
        )
        s = ResultSummary(run_id="r1", results=results)
        assert s.total == 2
        assert s.succeeded == 2
        assert s.failed == 0
        assert s.success_rate == 1.0
        assert s.all_succeeded
        assert not s.has_failures

    def test_mixed_results(self):
        results = (
            make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1"),
            make_result("a2", "t2", ResultStatus.FAILED, run_id="r1"),
            make_result("a3", "t3", ResultStatus.PARTIAL, run_id="r1"),
        )
        s = ResultSummary(run_id="r1", results=results)
        assert s.total == 3
        assert s.succeeded == 1
        assert s.failed == 1
        assert s.partial == 1
        assert s.success_rate == pytest.approx(1 / 3)
        assert s.has_failures
        assert s.has_partial
        assert not s.all_succeeded

    def test_total_cost(self):
        results = (
            make_result("a1", "t1", run_id="r1", cost=0.05),
            make_result("a2", "t2", run_id="r1", cost=0.03),
        )
        s = ResultSummary(run_id="r1", results=results)
        assert s.total_cost_usd == pytest.approx(0.08)

    def test_total_tokens(self):
        results = (
            make_result("a1", "t1", run_id="r1", input_tokens=200, output_tokens=100),
            make_result("a2", "t2", run_id="r1", input_tokens=300, output_tokens=150),
        )
        s = ResultSummary(run_id="r1", results=results)
        assert s.total_tokens == 750


# -- ResultCollector: record --


class TestCollectorRecord:
    """结果记录。"""

    def test_record_single(self):
        c = ResultCollectorImpl()
        r = make_result(run_id="r1")
        c.record(r)
        assert c.total_results == 1

    def test_record_duplicate_raises(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        with pytest.raises(ResultAlreadyExistsError):
            c.record(make_result("a1", "t1", run_id="r1"))

    def test_record_different_tasks_same_run(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r1"))
        assert c.total_results == 2

    def test_record_same_task_different_runs(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a1", "t1", run_id="r2"))
        assert c.total_results == 2

    def test_record_without_run_id(self):
        c = ResultCollectorImpl()
        r = AgentResult(agent_id="a1", task_id="t1", status=ResultStatus.SUCCESS)
        c.record(r)
        assert c.get("t1") is not None


# -- ResultCollector: query --


class TestCollectorQuery:
    """结果查询。"""

    def test_get_by_task_id(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        r = c.get("t1", "r1")
        assert r is not None
        assert r.agent_id == "a1"

    def test_get_not_found_returns_none(self):
        c = ResultCollectorImpl()
        assert c.get("nonexistent") is None

    def test_get_or_raise_not_found(self):
        c = ResultCollectorImpl()
        with pytest.raises(ResultNotFoundError):
            c.get_or_raise("nonexistent")

    def test_get_all(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r1"))
        c.record(make_result("a3", "t3", run_id="r2"))
        assert len(c.get_all("r1")) == 2
        assert len(c.get_all("r2")) == 1
        assert len(c.get_all("r3")) == 0

    def test_get_by_agent(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a1", "t2", run_id="r1"))
        c.record(make_result("a2", "t3", run_id="r1"))
        assert len(c.get_by_agent("a1", "r1")) == 2
        assert len(c.get_by_agent("a2", "r1")) == 1
        assert len(c.get_by_agent("a3", "r1")) == 0

    def test_get_by_agent_cross_runs(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a1", "t2", run_id="r2"))
        assert len(c.get_by_agent("a1")) == 2

    def test_get_by_status(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1"))
        c.record(make_result("a2", "t2", ResultStatus.FAILED, run_id="r1"))
        c.record(make_result("a3", "t3", ResultStatus.PARTIAL, run_id="r1"))
        assert len(c.get_by_status(ResultStatus.SUCCESS, "r1")) == 1
        assert len(c.get_by_status(ResultStatus.FAILED, "r1")) == 1
        assert len(c.get_by_status(ResultStatus.PARTIAL, "r1")) == 1

    def test_get_artifacts_for_task(self):
        c = ResultCollectorImpl()
        arts = (make_artifact("out.txt"), make_artifact("diff.patch", ArtifactKind.DIFF))
        c.record(make_result("a1", "t1", run_id="r1", artifacts=arts))
        result = c.get_artifacts("r1", "t1")
        assert len(result) == 2
        assert result[0].path == "out.txt"

    def test_get_artifacts_for_run(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1", artifacts=(make_artifact("a.txt"),)))
        c.record(make_result("a2", "t2", run_id="r1", artifacts=(make_artifact("b.txt"),)))
        all_arts = c.get_artifacts("r1")
        assert len(all_arts) == 2

    def test_get_artifacts_empty(self):
        c = ResultCollectorImpl()
        assert c.get_artifacts("r1") == ()
        c.record(make_result("a1", "t1", run_id="r1"))
        assert c.get_artifacts("r1", "t1") == ()


# -- ResultCollector: update --


class TestCollectorUpdate:
    """结果更新（部分 → 最终）。"""

    def test_update_existing(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.PARTIAL, run_id="r1"))
        c.update(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1"))
        r = c.get("t1", "r1")
        assert r.is_success

    def test_update_nonexistent_acts_as_record(self):
        c = ResultCollectorImpl()
        c.update(make_result("a1", "t1", run_id="r1"))
        assert c.total_results == 1

    def test_partial_to_success_flow(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.PARTIAL, run_id="r1", summary="wip"))
        assert c.get("t1", "r1").is_partial
        c.update(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1", summary="done"))
        r = c.get("t1", "r1")
        assert r.is_success
        assert r.summary == "done"


# -- ResultCollector: summarize --


class TestCollectorSummarize:
    """结果汇总。"""

    def test_summarize_empty(self):
        c = ResultCollectorImpl()
        s = c.summarize("r1")
        assert s.total == 0
        assert s.success_rate == 0.0

    def test_summarize_all_success(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r1"))
        s = c.summarize("r1")
        assert s.all_succeeded
        assert s.success_rate == 1.0

    def test_summarize_with_failures(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1"))
        c.record(make_result("a2", "t2", ResultStatus.FAILED, run_id="r1", error="err"))
        s = c.summarize("r1")
        assert s.has_failures
        assert s.succeeded == 1
        assert s.failed == 1

    def test_summarize_with_partial(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1"))
        c.record(make_result("a2", "t2", ResultStatus.PARTIAL, run_id="r1"))
        c.record(make_result("a3", "t3", ResultStatus.FAILED, run_id="r1"))
        s = c.summarize("r1")
        assert s.has_partial
        assert s.partial == 1

    def test_summarize_cost_and_tokens(self):
        c = ResultCollectorImpl()
        c.record(make_result(
            "a1", "t1", run_id="r1", cost=0.02, input_tokens=100, output_tokens=50,
        ))
        c.record(make_result(
            "a2", "t2", run_id="r1", cost=0.04, input_tokens=200, output_tokens=100,
        ))
        s = c.summarize("r1")
        assert s.total_cost_usd == pytest.approx(0.06)
        assert s.total_tokens == 450


# -- ResultCollector: clear --


class TestCollectorClear:
    """结果清除。"""

    def test_clear_single_run(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r2"))
        c.clear("r1")
        assert len(c.get_all("r1")) == 0
        assert len(c.get_all("r2")) == 1

    def test_clear_all(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r2"))
        c.clear_all()
        assert c.total_results == 0

    def test_clear_nonexistent_no_error(self):
        c = ResultCollectorImpl()
        c.clear("nonexistent")

    def test_run_ids_property(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        c.record(make_result("a2", "t2", run_id="r2"))
        assert set(c.run_ids) == {"r1", "r2"}


# -- ResultCollector: serialization --


class TestCollectorSerialization:
    """序列化。"""

    def test_to_dict(self):
        c = ResultCollectorImpl()
        arts = (make_artifact("out.txt", ArtifactKind.FILE),)
        c.record(make_result("a1", "t1", run_id="r1", artifacts=arts))
        d = c.to_dict("r1")
        assert d["run_id"] == "r1"
        assert d["total"] == 1
        assert len(d["results"]) == 1
        assert d["results"][0]["agent_id"] == "a1"
        assert d["results"][0]["status"] == "success"
        assert len(d["results"][0]["artifacts"]) == 1

    def test_to_dict_empty(self):
        c = ResultCollectorImpl()
        d = c.to_dict("r1")
        assert d["total"] == 0
        assert d["results"] == []

    def test_from_results(self):
        results = [
            make_result("a1", "t1", run_id="r1"),
            make_result("a2", "t2", run_id="r1"),
        ]
        c = ResultCollectorImpl.from_results(results)
        assert c.total_results == 2
        assert c.get("t1", "r1") is not None


# -- Protocol conformance --


class TestProtocolConformance:
    """异步 collect 接口。"""

    async def test_collect_returns_summary(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", run_id="r1"))
        s = await c.collect("r1")
        assert isinstance(s, ResultSummary)
        assert s.total == 1

    async def test_collect_empty_run(self):
        c = ResultCollectorImpl()
        s = await c.collect("r1")
        assert s.total == 0


# -- 端到端 --


class TestEndToEnd:
    """端到端：多 Agent 结果汇总。"""

    async def test_multi_agent_summary(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1", cost=0.01))
        c.record(make_result("a2", "t2", ResultStatus.SUCCESS, run_id="r1", cost=0.02))
        c.record(make_result("a3", "t3", ResultStatus.PARTIAL, run_id="r1", cost=0.03))
        c.record(make_result("a4", "t4", ResultStatus.FAILED, run_id="r1", error="boom", cost=0.0))
        c.record(make_result("a5", "t5", ResultStatus.TIMEOUT, run_id="r1", cost=0.0))

        s = await c.collect("r1")
        assert s.total == 5
        assert s.succeeded == 2
        assert s.partial == 1
        assert s.failed == 2
        assert s.has_failures
        assert s.has_partial
        assert not s.all_succeeded
        assert s.total_cost_usd == pytest.approx(0.06)

    async def test_artifact_collection_across_agents(self):
        c = ResultCollectorImpl()
        c.record(make_result(
            "a1", "t1", run_id="r1",
            artifacts=(
                make_artifact("a1_out.txt"),
                make_artifact("a1_diff.patch", ArtifactKind.DIFF),
            ),
        ))
        c.record(make_result(
            "a2", "t2", run_id="r1",
            artifacts=(make_artifact("a2_out.txt"),),
        ))

        all_arts = c.get_artifacts("r1")
        assert len(all_arts) == 3
        diffs = [a for a in all_arts if a.kind == ArtifactKind.DIFF]
        assert len(diffs) == 1

    async def test_partial_to_success_lifecycle(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.PARTIAL, run_id="r1", summary="wip"))
        s1 = c.summarize("r1")
        assert s1.partial == 1
        assert not s1.all_succeeded

        c.update(make_result("a1", "t1", ResultStatus.SUCCESS, run_id="r1", summary="done"))
        s2 = c.summarize("r1")
        assert s2.all_succeeded

    async def test_failed_results_queryable(self):
        c = ResultCollectorImpl()
        c.record(make_result("a1", "t1", ResultStatus.FAILED, run_id="r1", error="err1"))
        c.record(make_result("a2", "t2", ResultStatus.TIMEOUT, run_id="r1"))
        c.record(make_result("a3", "t3", ResultStatus.SUCCESS, run_id="r1"))

        failures = c.get_by_status(ResultStatus.FAILED, "r1")
        timeouts = c.get_by_status(ResultStatus.TIMEOUT, "r1")
        successes = c.get_by_status(ResultStatus.SUCCESS, "r1")
        assert len(failures) == 1
        assert len(timeouts) == 1
        assert len(successes) == 1
        assert failures[0].error == "err1"
