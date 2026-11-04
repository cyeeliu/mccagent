"""IntegrationManager 代码集成管理测试。

覆盖：
- 类型定义与校验（IntegrationRequest, DiffArtifact, MergeResult 等）
- 变更集收集（collect_diff）
- 提交（commit）
- 合并（无冲突、冲突、不同策略）
- 验证（verify）
- 回滚（rollback）
- 完整管线（integrate：成功、冲突、验证失败、空变更）
- 集成记录管理
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from mcc_harness.integration import (
    ConflictError,
    ConflictReport,
    DiffArtifact,
    IntegrationError,
    IntegrationManagerImpl,
    IntegrationNotFoundError,
    IntegrationRequest,
    IntegrationStatus,
    MergeResult,
    MergeStrategy,
    VerificationError,
    VerificationResult,
)

# -- 辅助函数与 fixture --


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """创建一个带初始 commit 的 git 仓库，返回路径。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "test@test.com"], repo)
    _git(["config", "user.name", "Test"], repo)
    _write(repo / "README.md", "# Repo\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)
    return repo


@pytest.fixture
def agent_worktree(git_repo: Path) -> tuple[Path, str]:
    """在 git_repo 中创建 agent worktree，返回 (worktree_path, branch_name)。"""
    branch = "task/agent-1"
    wt = git_repo / ".worktrees" / "agent-1"
    _git(["worktree", "add", "-b", branch, str(wt)], git_repo)
    return wt, branch


def make_request(
    worktree: Path,
    *,
    agent_id: str = "a1",
    task_id: str = "t1",
    target_branch: str = "main",
    agent_branch: str = "",
    commit_message: str = "",
) -> IntegrationRequest:
    return IntegrationRequest(
        agent_id=agent_id,
        task_id=task_id,
        worktree_dir=worktree,
        target_branch=target_branch,
        agent_branch=agent_branch,
        commit_message=commit_message,
    )


# -- IntegrationRequest --


class TestIntegrationRequest:
    """集成请求类型校验。"""

    def test_valid_request(self, tmp_path: Path):
        req = IntegrationRequest(
            agent_id="a1",
            task_id="t1",
            worktree_dir=tmp_path,
        )
        assert req.agent_id == "a1"
        assert req.target_branch == "main"

    def test_empty_agent_id_raises(self, tmp_path: Path):
        with pytest.raises(IntegrationError, match="agent_id"):
            IntegrationRequest(agent_id="", task_id="t1", worktree_dir=tmp_path)

    def test_empty_task_id_raises(self, tmp_path: Path):
        with pytest.raises(IntegrationError, match="task_id"):
            IntegrationRequest(agent_id="a1", task_id="", worktree_dir=tmp_path)

    def test_defaults(self, tmp_path: Path):
        req = IntegrationRequest(
            agent_id="a1", task_id="t1", worktree_dir=tmp_path,
        )
        assert req.target_branch == "main"
        assert req.agent_branch == ""
        assert req.commit_message == ""
        assert req.team_run_id is None
        assert req.trace_id is None


# -- DiffArtifact --


class TestDiffArtifact:
    """变更集类型。"""

    def test_empty_diff(self):
        diff = DiffArtifact(diff_content="")
        assert diff.is_empty
        assert diff.total_files == 0

    def test_non_empty_diff(self):
        diff = DiffArtifact(
            diff_content="@@ -1 +1 @@",
            files_changed=("a.py",),
            untracked_files=("b.py",),
            insertions=5,
            deletions=2,
        )
        assert not diff.is_empty
        assert diff.total_files == 2

    def test_only_untracked(self):
        diff = DiffArtifact(
            diff_content="",
            untracked_files=("new.py",),
        )
        assert not diff.is_empty
        assert diff.total_files == 1

    @pytest.mark.parametrize("ins,dels,expected", [
        (10, 5, 15),
        (0, 0, 0),
        (100, 50, 150),
    ])
    def test_total_files_with_changes(self, ins, dels, expected):
        diff = DiffArtifact(
            diff_content="x",
            files_changed=("a.py",),
            insertions=ins,
            deletions=dels,
        )
        assert diff.total_files == 1


# -- MergeResult / ConflictReport / VerificationResult --


class TestMergeResult:
    """合并结果类型。"""

    def test_success_result(self):
        r = MergeResult(status=IntegrationStatus.MERGED, merged_commit="abc123")
        assert r.is_success
        assert not r.is_conflict
        assert not r.is_rolled_back

    def test_conflict_result(self):
        r = MergeResult(
            status=IntegrationStatus.CONFLICT,
            conflicts=ConflictReport(conflicted_files=("a.py",)),
        )
        assert r.is_conflict
        assert not r.is_success

    def test_rolled_back_result(self):
        r = MergeResult(status=IntegrationStatus.ROLLED_BACK)
        assert r.is_rolled_back

    def test_completed_result(self):
        r = MergeResult(status=IntegrationStatus.COMPLETED)
        assert r.is_completed


class TestConflictReport:
    """冲突报告类型。"""

    def test_no_conflicts(self):
        report = ConflictReport()
        assert not report.has_conflicts

    def test_with_conflicts(self):
        report = ConflictReport(
            conflicted_files=("a.py", "b.py"),
            details={"a.py": "<<<<<<<", "b.py": ">>>>>>>"},
        )
        assert report.has_conflicts
        assert len(report.conflicted_files) == 2


class TestVerificationResult:
    """验证结果类型。"""

    def test_passed(self):
        v = VerificationResult(passed=True)
        assert not v.has_failures

    def test_failed(self):
        v = VerificationResult(
            passed=False,
            failures=("test failed",),
            exit_code=1,
        )
        assert v.has_failures
        assert len(v.failures) == 1


# -- IntegrationStatus / MergeStrategy --


class TestEnums:
    """枚举值。"""

    @pytest.mark.parametrize("status,expected", [
        (IntegrationStatus.PENDING, "pending"),
        (IntegrationStatus.COLLECTED, "collected"),
        (IntegrationStatus.COMMITTED, "committed"),
        (IntegrationStatus.MERGED, "merged"),
        (IntegrationStatus.CONFLICT, "conflict"),
        (IntegrationStatus.VALIDATION_FAILED, "validation_failed"),
        (IntegrationStatus.ROLLED_BACK, "rolled_back"),
        (IntegrationStatus.COMPLETED, "completed"),
    ])
    def test_integration_status_values(self, status, expected):
        assert status == expected

    @pytest.mark.parametrize("strategy,expected", [
        (MergeStrategy.FAST_FORWARD, "fast_forward"),
        (MergeStrategy.MERGE_COMMIT, "merge_commit"),
        (MergeStrategy.SQUASH, "squash"),
    ])
    def test_merge_strategy_values(self, strategy, expected):
        assert strategy == expected


# -- collect_diff --


class TestCollectDiff:
    """变更集收集。"""

    def test_clean_worktree(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        req = make_request(wt)
        diff = mgr.collect_diff(req)
        assert diff.is_empty
        assert diff.base_commit != ""

    def test_modified_file(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "README.md", "# Modified\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        assert not diff.is_empty
        assert "README.md" in diff.files_changed
        assert diff.insertions > 0

    def test_new_file(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "new.py", "print('hello')\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        assert not diff.is_empty
        assert "new.py" in diff.untracked_files

    def test_multiple_changes(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "README.md", "# Changed\n")
        _write(wt / "a.py", "x = 1\n")
        _write(wt / "b.py", "y = 2\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        assert diff.total_files >= 3

    def test_nonexistent_worktree(self, tmp_path):
        mgr = IntegrationManagerImpl()
        with pytest.raises(IntegrationError, match="worktree not found"):
            mgr.collect_diff(make_request(tmp_path / "nope"))

    def test_non_git_dir(self, tmp_path):
        mgr = IntegrationManagerImpl()
        with pytest.raises(IntegrationError, match="not a git repo"):
            mgr.collect_diff(make_request(tmp_path))


# -- commit --


class TestCommit:
    """提交变更。"""

    def test_commit_modified_file(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "README.md", "# Modified\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        req = make_request(wt, commit_message="update readme")
        sha = mgr.commit(req, diff)
        assert len(sha) == 40

    def test_commit_new_file(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        sha = mgr.commit(make_request(wt), diff)
        assert len(sha) == 40

    def test_commit_empty_diff_raises(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        with pytest.raises(IntegrationError, match="nothing to commit"):
            mgr.commit(make_request(wt), diff)

    def test_commit_default_message(self, agent_worktree):
        wt, _ = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt), diff)
        log = _git(["log", "-1", "--format=%s"], wt)
        assert "t1" in log and "a1" in log


# -- merge --


class TestMerge:
    """合并操作。"""

    def test_merge_no_conflict(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt, commit_message="add new.py"), diff)
        result = mgr.merge(
            make_request(wt, agent_branch=branch, commit_message="merge"),
            MergeStrategy.MERGE_COMMIT,
        )
        assert result.is_success
        assert result.merged_commit is not None

    def test_merge_with_conflict(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "README.md", "# Agent Change\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt, commit_message="agent change"), diff)

        _write(git_repo / "README.md", "# Main Change\n")
        _git(["add", "-A"], git_repo)
        _git(["commit", "-m", "main change"], git_repo)

        result = mgr.merge(
            make_request(wt, agent_branch=branch),
            MergeStrategy.MERGE_COMMIT,
        )
        assert result.is_conflict
        assert result.conflicts is not None
        assert result.conflicts.has_conflicts
        assert "README.md" in result.conflicts.conflicted_files

    def test_fast_forward_merge(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt, commit_message="add new.py"), diff)
        result = mgr.merge(
            make_request(wt, agent_branch=branch),
            MergeStrategy.FAST_FORWARD,
        )
        assert result.is_success

    def test_squash_merge(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt, commit_message="add new.py"), diff)
        result = mgr.merge(
            make_request(wt, agent_branch=branch, commit_message="squash"),
            MergeStrategy.SQUASH,
        )
        assert result.is_success


# -- verify --


class TestVerify:
    """验证操作。"""

    def test_verify_pass(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        result = mgr.verify(make_request(wt), ["true"])
        assert result.passed
        assert result.exit_code == 0

    def test_verify_fail(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        result = mgr.verify(make_request(wt), ["false"])
        assert not result.passed
        assert result.exit_code != 0
        assert len(result.failures) > 0

    def test_verify_with_output(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        result = mgr.verify(
            make_request(wt),
            ["python3", "-c", "print('all ok')"],
        )
        assert result.passed
        assert "all ok" in result.output

    def test_verify_command_not_found(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        with pytest.raises(VerificationError, match="command not found"):
            mgr.verify(make_request(wt), ["nonexistent-cmd-xyz"])


# -- rollback --


class TestRollback:
    """回滚操作。"""

    def test_rollback_to_previous(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        base = _git(["rev-parse", "HEAD"], git_repo)

        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        diff = mgr.collect_diff(make_request(wt))
        mgr.commit(make_request(wt, commit_message="add"), diff)
        mgr.merge(
            make_request(wt, agent_branch=branch, commit_message="merge"),
        )

        result = mgr.rollback(make_request(wt, agent_branch=branch), base)
        assert result.is_rolled_back
        assert result.merged_commit == base

        current = _git(["rev-parse", "HEAD"], git_repo)
        assert current == base


# -- integrate (full pipeline) --


class TestIntegrate:
    """完整集成管线。"""

    def test_integrate_success(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add new.py"),
        )
        assert result.is_completed
        assert result.merged_commit is not None
        assert result.diff is not None

    def test_integrate_empty_diff(self, agent_worktree):
        wt, _ = agent_worktree
        mgr = IntegrationManagerImpl()
        result = mgr.integrate(make_request(wt))
        assert result.is_completed
        assert result.diff is not None
        assert result.diff.is_empty

    def test_integrate_with_conflict(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "README.md", "# Agent\n")

        _write(git_repo / "README.md", "# Main\n")
        _git(["add", "-A"], git_repo)
        _git(["commit", "-m", "main"], git_repo)

        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="agent change"),
        )
        assert result.is_conflict
        assert result.conflicts is not None
        assert result.conflicts.has_conflicts

    def test_integrate_verify_pass(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add"),
            verify_command=["true"],
        )
        assert result.is_completed
        assert result.verification is not None
        assert result.verification.passed

    def test_integrate_verify_fail_rolls_back(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        base = _git(["rev-parse", "HEAD"], git_repo)

        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add"),
            verify_command=["false"],
        )
        assert result.status == IntegrationStatus.VALIDATION_FAILED
        assert result.verification is not None
        assert not result.verification.passed

        current = _git(["rev-parse", "HEAD"], git_repo)
        assert current == base

    def test_integrate_records_history(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add"),
        )
        assert mgr.has_record("a1", "t1")
        record = mgr.get_record("a1", "t1")
        assert record.is_completed


# -- 记录管理 --


class TestRecordManagement:
    """集成记录管理。"""

    def test_get_record_not_found(self):
        mgr = IntegrationManagerImpl()
        with pytest.raises(IntegrationNotFoundError):
            mgr.get_record("a1", "t1")

    def test_has_record_false(self):
        mgr = IntegrationManagerImpl()
        assert not mgr.has_record("a1", "t1")

    def test_clear(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add"),
        )
        assert mgr.total_records > 0
        mgr.clear()
        assert mgr.total_records == 0

    def test_multiple_records(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add1"),
        )
        _write(wt / "new2.py", "y = 2\n")
        mgr.integrate(
            make_request(
                wt, agent_id="a2", task_id="t2",
                agent_branch=branch, commit_message="add2",
            ),
        )
        assert mgr.total_records == 2
        assert mgr.has_record("a1", "t1")
        assert mgr.has_record("a2", "t2")


# -- 端到端 --


class TestEndToEnd:
    """端到端多 Agent 集成场景。"""

    def test_two_agents_no_conflict(self, git_repo):
        wt1 = git_repo / ".worktrees" / "agent-1"
        wt2 = git_repo / ".worktrees" / "agent-2"
        _git(["worktree", "add", "-b", "task/agent-1", str(wt1)], git_repo)
        _git(["worktree", "add", "-b", "task/agent-2", str(wt2)], git_repo)

        _write(wt1 / "a.py", "x = 1\n")
        _write(wt2 / "b.py", "y = 2\n")

        mgr = IntegrationManagerImpl()
        r1 = mgr.integrate(
            make_request(
                wt1, agent_id="a1", task_id="t1",
                agent_branch="task/agent-1", commit_message="add a.py",
            ),
        )
        assert r1.is_completed

        r2 = mgr.integrate(
            make_request(
                wt2, agent_id="a2", task_id="t2",
                agent_branch="task/agent-2", commit_message="add b.py",
            ),
        )
        assert r2.is_completed

        assert (git_repo / "a.py").exists()
        assert (git_repo / "b.py").exists()

    def test_rollback_preserves_worktree(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        base = _git(["rev-parse", "HEAD"], git_repo)

        _write(wt / "new.py", "x = 1\n")
        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="add"),
            verify_command=["false"],
        )
        assert result.status == IntegrationStatus.VALIDATION_FAILED

        current = _git(["rev-parse", "HEAD"], git_repo)
        assert current == base

        assert (wt / "new.py").exists()

    def test_conflict_preserves_worktree(self, git_repo, agent_worktree):
        wt, branch = agent_worktree
        _write(wt / "README.md", "# Agent\n")

        _write(git_repo / "README.md", "# Main\n")
        _git(["add", "-A"], git_repo)
        _git(["commit", "-m", "main"], git_repo)

        mgr = IntegrationManagerImpl()
        result = mgr.integrate(
            make_request(wt, agent_branch=branch, commit_message="agent"),
        )
        assert result.is_conflict

        agent_readme = (wt / "README.md").read_text()
        assert "Agent" in agent_readme
