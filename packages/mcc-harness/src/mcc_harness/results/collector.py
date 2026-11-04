"""结果收集器实现：记录、查询与汇总 Agent 执行结果。

提供结构化结果存储，支持按 run_id / task_id / agent_id 查询，
聚合生成 ResultSummary（区分成功、部分成功和失败）。
实现 ResultCollector Protocol（team.orchestrator）的异步接口，
可直接注入 TeamOrchestrator 替换默认 SimpleResultCollector。

设计约束：
- 自然语言 summary 不替代结构化 status。
- 大文件产物通过 ArtifactRef 引用，不内联内容。
- 结果写入支持 update 覆盖（部分结果 → 最终结果）。
"""

from __future__ import annotations

from collections.abc import Iterable

from .base import (
    AgentResult,
    ArtifactRef,
    ResultAlreadyExistsError,
    ResultNotFoundError,
    ResultStatus,
    ResultSummary,
)


class ResultCollectorImpl:
    """结果收集器：记录、查询与汇总 Agent 执行结果。

    内部按 run_id → task_id → AgentResult 二级索引存储。
    record() 首次写入，update() 覆盖更新（支持部分→最终流转）。
    summarize() 聚合为 ResultSummary，区分成功/部分/失败。
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, AgentResult]] = {}

    def record(self, result: AgentResult) -> None:
        """记录执行结果。

        首次写入，重复记录同一 (run_id, task_id) 抛 ResultAlreadyExistsError。
        若 result 未设置 team_run_id，使用 "_default" 作为存储分区。
        """
        run_id = result.team_run_id or "_default"
        bucket = self._store.setdefault(run_id, {})
        if result.task_id in bucket:
            raise ResultAlreadyExistsError(
                f"result already exists: run={run_id} task={result.task_id}",
            )
        bucket[result.task_id] = result

    def update(self, result: AgentResult) -> None:
        """更新已存在的执行结果。

        用于部分结果 → 最终结果的流转。
        若结果不存在则等同 record()。
        """
        run_id = result.team_run_id or "_default"
        bucket = self._store.setdefault(run_id, {})
        bucket[result.task_id] = result

    def get(
        self,
        task_id: str,
        run_id: str | None = None,
    ) -> AgentResult | None:
        """按 task_id 查询结果，未找到返回 None。"""
        if run_id is not None:
            bucket = self._store.get(run_id)
            if bucket is None:
                return None
            return bucket.get(task_id)
        for bucket in self._store.values():
            result = bucket.get(task_id)
            if result is not None:
                return result
        return None

    def get_or_raise(
        self,
        task_id: str,
        run_id: str | None = None,
    ) -> AgentResult:
        """按 task_id 查询结果，未找到抛出 ResultNotFoundError。"""
        result = self.get(task_id, run_id)
        if result is None:
            raise ResultNotFoundError(
                f"result not found: task={task_id} run={run_id}",
            )
        return result

    def get_all(self, run_id: str) -> tuple[AgentResult, ...]:
        """获取 run_id 下全部结果。"""
        bucket = self._store.get(run_id, {})
        return tuple(bucket.values())

    def get_by_agent(
        self,
        agent_id: str,
        run_id: str | None = None,
    ) -> tuple[AgentResult, ...]:
        """按 agent_id 查询结果。"""
        if run_id is not None:
            bucket = self._store.get(run_id, {})
            return tuple(
                r for r in bucket.values() if r.agent_id == agent_id
            )
        results: list[AgentResult] = []
        for bucket in self._store.values():
            results.extend(
                r for r in bucket.values() if r.agent_id == agent_id
            )
        return tuple(results)

    def get_by_status(
        self,
        status: ResultStatus,
        run_id: str | None = None,
    ) -> tuple[AgentResult, ...]:
        """按状态过滤结果。"""
        if run_id is not None:
            bucket = self._store.get(run_id, {})
            return tuple(
                r for r in bucket.values() if r.status == status
            )
        results: list[AgentResult] = []
        for bucket in self._store.values():
            results.extend(
                r for r in bucket.values() if r.status == status
            )
        return tuple(results)

    def get_artifacts(
        self,
        run_id: str,
        task_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """获取产物引用列表。

        task_id 为 None 时返回 run_id 下全部产物。
        """
        if task_id is not None:
            result = self.get(task_id, run_id)
            if result is None:
                return ()
            return result.artifacts
        artifacts: list[ArtifactRef] = []
        for result in self.get_all(run_id):
            artifacts.extend(result.artifacts)
        return tuple(artifacts)

    def summarize(self, run_id: str) -> ResultSummary:
        """聚合 run_id 下全部结果为 ResultSummary。"""
        return ResultSummary(
            run_id=run_id,
            results=self.get_all(run_id),
        )

    def clear(self, run_id: str) -> None:
        """清除 run_id 下全部结果。"""
        self._store.pop(run_id, None)

    def clear_all(self) -> None:
        """清除全部结果。"""
        self._store.clear()

    @property
    def run_ids(self) -> tuple[str, ...]:
        """有结果记录的 run_id 列表。"""
        return tuple(
            rid for rid, bucket in self._store.items() if bucket
        )

    @property
    def total_results(self) -> int:
        """全部结果总数。"""
        return sum(len(bucket) for bucket in self._store.values())

    async def collect(self, run_id: str) -> ResultSummary:
        """异步收集接口：返回 ResultSummary。

        兼容 ResultCollector Protocol 的异步调用模式。
        """
        return self.summarize(run_id)

    def to_dict(self, run_id: str) -> dict[str, object]:
        """序列化 run_id 下结果为可 JSON 化字典。"""
        results = self.get_all(run_id)
        return {
            "run_id": run_id,
            "results": [_result_to_dict(r) for r in results],
            "total": len(results),
        }

    @classmethod
    def from_results(cls, results: Iterable[AgentResult]) -> ResultCollectorImpl:
        """从已有结果列表构建收集器。"""
        collector = cls()
        for result in results:
            collector.update(result)
        return collector


def _result_to_dict(result: AgentResult) -> dict[str, object]:
    """将 AgentResult 序列化为字典。"""
    return {
        "agent_id": result.agent_id,
        "task_id": result.task_id,
        "status": result.status.value,
        "summary": result.summary,
        "error": result.error,
        "metrics": {
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "cache_read_tokens": result.metrics.cache_read_tokens,
            "cache_write_tokens": result.metrics.cache_write_tokens,
            "cost_usd": result.metrics.cost_usd,
            "started_at": result.metrics.started_at,
            "finished_at": result.metrics.finished_at,
        },
        "artifacts": [
            {
                "path": a.path,
                "kind": a.kind.value,
                "description": a.description,
                "size_bytes": a.size_bytes,
                "checksum": a.checksum,
            }
            for a in result.artifacts
        ],
        "team_run_id": result.team_run_id,
        "trace_id": result.trace_id,
    }
