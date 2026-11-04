"""代码集成管理类型定义：变更集、合并结果与冲突报告。

将 Agent 工作区成果的收集、审查、提交、合并、冲突解决和回滚
集中定义为独立领域服务，与 Teammate 生命周期解耦。

通过 IntegrationRequest 描述一次集成请求，
通过 DiffArtifact 描述可审查变更集，
通过 MergeResult 描述合并结果（含冲突与回滚状态）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class IntegrationError(RuntimeError):
    """代码集成基础错误。"""


class IntegrationNotFoundError(IntegrationError):
    """集成记录不存在。"""


class ConflictError(IntegrationError):
    """合并冲突错误。"""


class VerificationError(IntegrationError):
    """验证执行错误。"""


class IntegrationStatus(StrEnum):
    """集成流程状态。"""

    PENDING = "pending"
    COLLECTED = "collected"
    COMMITTED = "committed"
    MERGED = "merged"
    CONFLICT = "conflict"
    VALIDATION_FAILED = "validation_failed"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"


class MergeStrategy(StrEnum):
    """合并策略。"""

    FAST_FORWARD = "fast_forward"
    MERGE_COMMIT = "merge_commit"
    SQUASH = "squash"


@dataclass(frozen=True)
class DiffArtifact:
    """可审查变更集：diff 内容与文件统计。

    files_changed 为有修改的已跟踪文件路径。
    untracked_files 为未跟踪文件路径。
    diff_content 为 ``git diff HEAD`` 的完整文本。
    insertions / deletions 为变更行数统计。
    """

    diff_content: str
    files_changed: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()
    insertions: int = 0
    deletions: int = 0
    base_commit: str = ""
    head_commit: str = ""

    @property
    def is_empty(self) -> bool:
        """变更集是否为空。"""
        return not self.diff_content.strip() and not self.untracked_files

    @property
    def total_files(self) -> int:
        """涉及文件总数（已修改 + 未跟踪）。"""
        return len(self.files_changed) + len(self.untracked_files)


@dataclass(frozen=True)
class ConflictReport:
    """合并冲突报告：定位到具体文件。

    conflicted_files 为冲突文件路径列表。
    details 为文件路径到冲突描述的映射。
    """

    conflicted_files: tuple[str, ...] = ()
    details: dict[str, str] = field(default_factory=dict)

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicted_files) > 0


@dataclass(frozen=True)
class VerificationResult:
    """验证结果：测试或检查通过与否。

    passed 为是否通过。
    failures 为失败信息列表。
    output 为验证命令的完整输出。
    """

    passed: bool
    failures: tuple[str, ...] = ()
    output: str = ""
    exit_code: int = 0

    @property
    def has_failures(self) -> bool:
        return not self.passed


@dataclass(frozen=True)
class MergeResult:
    """合并结果：状态 + 合并 commit + 冲突 + 验证。

    status 为集成流程状态。
    merged_commit 为合并后的 commit SHA（成功时）。
    conflicts 为冲突报告（冲突时）。
    verification 为验证结果（验证后）。
    diff 为变更集快照。
    error 为错误描述（失败时）。
    """

    status: IntegrationStatus
    merged_commit: str | None = None
    conflicts: ConflictReport | None = None
    verification: VerificationResult | None = None
    diff: DiffArtifact | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == IntegrationStatus.MERGED

    @property
    def is_conflict(self) -> bool:
        return self.status == IntegrationStatus.CONFLICT

    @property
    def is_rolled_back(self) -> bool:
        return self.status == IntegrationStatus.ROLLED_BACK

    @property
    def is_completed(self) -> bool:
        return self.status == IntegrationStatus.COMPLETED


@dataclass(frozen=True)
class IntegrationRequest:
    """集成请求：关联 Agent 工作区与目标分支。

    agent_id / task_id 关联执行主体。
    worktree_dir 为 Agent 工作区路径（git worktree）。
    target_branch 为合并目标分支（默认 main）。
    agent_branch 为 Agent 工作分支（默认从 worktree 推断）。
    commit_message 为提交消息。
    team_run_id / trace_id 用于跨组件关联。
    """

    agent_id: str
    task_id: str
    worktree_dir: Path
    target_branch: str = "main"
    agent_branch: str = ""
    commit_message: str = ""
    team_run_id: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise IntegrationError("agent_id must not be empty")
        if not self.task_id:
            raise IntegrationError("task_id must not be empty")
