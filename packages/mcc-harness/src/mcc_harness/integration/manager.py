"""代码集成管理实现：收集、提交、合并、冲突解决与回滚。

管理 Agent 工作区成果的集成生命周期：
1. collect_diff — 收集 diff、未跟踪文件和统计信息
2. commit — 在 Agent 分支上提交变更
3. merge — 将 Agent 分支合并到目标分支
4. verify — 集成后运行验证（如测试）
5. rollback — 验证失败时回滚到基准 commit
6. integrate — 完整管线（collect → commit → merge → verify → rollback）

设计约束：
- Agent 完成不等于代码集成完成。
- 未经检查的 diff 不得自动合并到主分支。
- 任何失败都不得丢失原始修改（冲突时保留工作区）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .base import (
    ConflictReport,
    DiffArtifact,
    IntegrationError,
    IntegrationNotFoundError,
    IntegrationRequest,
    IntegrationStatus,
    MergeResult,
    MergeStrategy,
    VerificationError,
    VerificationResult,
)


def _run_git(
    args: list[str],
    cwd: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """执行 git 命令，返回结果。

    check 为 True 时 git 失败抛 IntegrationError。
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip() if e.stderr else str(e)
        raise IntegrationError(f"git {' '.join(args)} failed: {err}") from e
    except FileNotFoundError as e:
        raise IntegrationError("git not found") from e


def _is_git_repo(cwd: Path) -> bool:
    """检查路径是否为 git 仓库。"""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


class IntegrationManagerImpl:
    """代码集成管理器：收集、提交、合并、验证与回滚。

    所有 git 操作通过 subprocess 执行，不依赖外部库。
    集成记录按 (agent_id, task_id) 索引，支持查询历史。
    """

    def __init__(self) -> None:
        self._records: dict[str, MergeResult] = {}

    @staticmethod
    def _key(agent_id: str, task_id: str) -> str:
        return f"{agent_id}:{task_id}"

    def collect_diff(self, request: IntegrationRequest) -> DiffArtifact:
        """从 Agent 工作区收集变更集。

        收集 ``git diff HEAD`` 和未跟踪文件列表。
        非 git 目录抛 IntegrationError。
        空变更集返回 is_empty=True。
        """
        wt = request.worktree_dir
        if not wt.exists():
            raise IntegrationError(f"worktree not found: {wt}")
        if not _is_git_repo(wt):
            raise IntegrationError(f"not a git repo: {wt}")

        base = _run_git(["rev-parse", "HEAD"], cwd=wt).stdout.strip()

        diff_result = _run_git(["diff", "HEAD"], cwd=wt, check=False)
        diff_content = diff_result.stdout

        untracked_result = _run_git(
            ["ls-files", "--others", "--exclude-standard"], cwd=wt,
        )
        untracked = tuple(
            line for line in untracked_result.stdout.strip().splitlines() if line
        )

        changed_result = _run_git(
            ["diff", "--name-only", "HEAD"], cwd=wt, check=False,
        )
        changed = tuple(
            line for line in changed_result.stdout.strip().splitlines() if line
        )

        stat_result = _run_git(
            ["diff", "--numstat", "HEAD"], cwd=wt, check=False,
        )
        insertions = 0
        deletions = 0
        for line in stat_result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                ins = parts[0]
                dels = parts[1]
                if ins != "-":
                    insertions += int(ins)
                if dels != "-":
                    deletions += int(dels)

        return DiffArtifact(
            diff_content=diff_content,
            files_changed=changed,
            untracked_files=untracked,
            insertions=insertions,
            deletions=deletions,
            base_commit=base,
            head_commit=base,
        )

    def commit(
        self,
        request: IntegrationRequest,
        diff: DiffArtifact,
    ) -> str:
        """在 Agent 工作区提交全部变更。

        暂存所有变更（含未跟踪文件）后提交。
        空变更集抛 IntegrationError。
        返回新 commit SHA。
        """
        if diff.is_empty:
            raise IntegrationError("nothing to commit: diff is empty")

        wt = request.worktree_dir
        _run_git(["add", "-A"], cwd=wt)

        message = request.commit_message or f"task/{request.task_id}: {request.agent_id}"
        _run_git(["commit", "-m", message], cwd=wt)

        return _run_git(["rev-parse", "HEAD"], cwd=wt).stdout.strip()

    def merge(
        self,
        request: IntegrationRequest,
        strategy: MergeStrategy = MergeStrategy.MERGE_COMMIT,
    ) -> MergeResult:
        """将 Agent 分支合并到目标分支。

        在目标仓库（worktree 的主仓库）执行合并。
        合并冲突时保留工作区并返回 ConflictReport。
        """
        wt = request.worktree_dir
        agent_branch = request.agent_branch or self._detect_branch(wt)
        target = request.target_branch

        repo_root = self._repo_root(wt)

        _run_git(["checkout", target], cwd=repo_root)

        merge_args = ["merge"]
        if strategy == MergeStrategy.FAST_FORWARD:
            merge_args.extend(["--ff-only", agent_branch])
        elif strategy == MergeStrategy.SQUASH:
            merge_args.extend(["--squash", agent_branch])
        else:
            merge_args.append(agent_branch)

        if strategy == MergeStrategy.SQUASH:
            merge_result = _run_git(merge_args, cwd=repo_root, check=False)
            if merge_result.returncode != 0:
                return self._build_conflict_result(
                    repo_root, request, merge_result,
                )
            _run_git(
                ["commit", "-m", request.commit_message or f"squash: {agent_branch}"],
                cwd=repo_root,
            )
        else:
            merge_result = _run_git(merge_args, cwd=repo_root, check=False)
            if merge_result.returncode != 0:
                return self._build_conflict_result(
                    repo_root, request, merge_result,
                )

        merged_commit = _run_git(
            ["rev-parse", "HEAD"], cwd=repo_root,
        ).stdout.strip()

        result = MergeResult(
            status=IntegrationStatus.MERGED,
            merged_commit=merged_commit,
        )
        self._records[self._key(request.agent_id, request.task_id)] = result
        return result

    def verify(
        self,
        request: IntegrationRequest,
        command: list[str],
    ) -> VerificationResult:
        """在目标仓库运行验证命令。

        command 为验证命令（如 ["python", "-m", "pytest"]）。
        返回 VerificationResult，含退出码和输出。
        """
        wt = request.worktree_dir
        repo_root = self._repo_root(wt)

        try:
            result = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as e:
            raise VerificationError(f"command not found: {command[0]}") from e

        output = result.stdout + result.stderr
        passed = result.returncode == 0
        failures: tuple[str, ...] = ()
        if not passed:
            fail_lines = [
                line for line in output.strip().splitlines()
                if "fail" in line.lower() or "error" in line.lower()
            ]
            failures = tuple(fail_lines[:20]) or ("verification failed",)

        return VerificationResult(
            passed=passed,
            failures=failures,
            output=output,
            exit_code=result.returncode,
        )

    def rollback(
        self,
        request: IntegrationRequest,
        target_commit: str,
    ) -> MergeResult:
        """回滚目标分支到指定 commit。

        使用 ``git reset --hard`` 回滚。
        返回 ROLLED_BACK 状态的 MergeResult。
        """
        wt = request.worktree_dir
        repo_root = self._repo_root(wt)

        _run_git(["reset", "--hard", target_commit], cwd=repo_root)

        result = MergeResult(
            status=IntegrationStatus.ROLLED_BACK,
            merged_commit=target_commit,
            meta={"rolled_back_from": request.agent_branch or self._detect_branch(wt)},
        )
        self._records[self._key(request.agent_id, request.task_id)] = result
        return result

    def integrate(
        self,
        request: IntegrationRequest,
        strategy: MergeStrategy = MergeStrategy.MERGE_COMMIT,
        verify_command: list[str] | None = None,
    ) -> MergeResult:
        """完整集成管线：collect → commit → merge → verify → rollback。

        1. 收集变更集
        2. 在 Agent 分支提交
        3. 合并到目标分支
        4. （可选）运行验证
        5. 验证失败则回滚到合并前 commit

        任何步骤失败都保留原始修改，不丢失工作区。
        """
        diff = self.collect_diff(request)
        if diff.is_empty:
            result = MergeResult(
                status=IntegrationStatus.COMPLETED,
                diff=diff,
                meta={"reason": "no changes"},
            )
            self._records[self._key(request.agent_id, request.task_id)] = result
            return result

        wt = request.worktree_dir
        repo_root = self._repo_root(wt)
        pre_merge_commit = _run_git(
            ["rev-parse", "HEAD"], cwd=repo_root,
        ).stdout.strip()

        try:
            self.commit(request, diff)
        except IntegrationError as e:
            result = MergeResult(
                status=IntegrationStatus.PENDING,
                diff=diff,
                error=str(e),
            )
            self._records[self._key(request.agent_id, request.task_id)] = result
            return result

        merge_result = self.merge(request, strategy)
        if merge_result.is_conflict:
            self._records[self._key(request.agent_id, request.task_id)] = merge_result
            return merge_result

        if verify_command is not None:
            verification = self.verify(request, verify_command)
            if verification.has_failures:
                rollback_result = self.rollback(request, pre_merge_commit)
                result = MergeResult(
                    status=IntegrationStatus.VALIDATION_FAILED,
                    merged_commit=pre_merge_commit,
                    verification=verification,
                    diff=diff,
                    meta={"rollback": rollback_result.meta},
                )
                self._records[self._key(request.agent_id, request.task_id)] = result
                return result

            final = MergeResult(
                status=IntegrationStatus.COMPLETED,
                merged_commit=merge_result.merged_commit,
                verification=verification,
                diff=diff,
            )
            self._records[self._key(request.agent_id, request.task_id)] = final
            return final

        final = MergeResult(
            status=IntegrationStatus.COMPLETED,
            merged_commit=merge_result.merged_commit,
            diff=diff,
        )
        self._records[self._key(request.agent_id, request.task_id)] = final
        return final

    def get_record(
        self,
        agent_id: str,
        task_id: str,
    ) -> MergeResult:
        """查询集成记录，不存在抛 IntegrationNotFoundError。"""
        key = self._key(agent_id, task_id)
        if key not in self._records:
            raise IntegrationNotFoundError(
                f"no integration record: agent={agent_id} task={task_id}",
            )
        return self._records[key]

    def has_record(self, agent_id: str, task_id: str) -> bool:
        """是否存在集成记录。"""
        return self._key(agent_id, task_id) in self._records

    def clear(self) -> None:
        """清除全部集成记录。"""
        self._records.clear()

    @property
    def total_records(self) -> int:
        """集成记录总数。"""
        return len(self._records)

    @staticmethod
    def _detect_branch(cwd: Path) -> str:
        """检测当前分支名。"""
        result = _run_git(
            ["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd,
        )
        return result.stdout.strip()

    @staticmethod
    def _repo_root(cwd: Path) -> Path:
        """获取主仓库根目录（非 linked worktree）。"""
        result = _run_git(["rev-parse", "--git-common-dir"], cwd=cwd)
        return Path(result.stdout.strip()).parent

    @staticmethod
    def _build_conflict_result(
        repo_root: Path,
        request: IntegrationRequest,
        merge_output: subprocess.CompletedProcess[str],
    ) -> MergeResult:
        """从合并失败输出构建冲突结果。"""
        diff_result = _run_git(["diff", "--name-only", "--diff-filter=U"], cwd=repo_root)
        conflicted = tuple(
            line for line in diff_result.stdout.strip().splitlines() if line
        )

        details: dict[str, str] = {}
        for f in conflicted:
            try:
                content = _run_git(
                    ["diff", f], cwd=repo_root, check=False,
                ).stdout
                details[f] = content
            except IntegrationError:
                details[f] = "conflict details unavailable"

        _run_git(["merge", "--abort"], cwd=repo_root, check=False)

        return MergeResult(
            status=IntegrationStatus.CONFLICT,
            conflicts=ConflictReport(
                conflicted_files=conflicted,
                details=details,
            ),
            error=merge_output.stderr.strip() or merge_output.stdout.strip(),
        )
