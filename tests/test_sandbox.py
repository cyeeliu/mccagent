"""沙箱测试：restrictions 档真执行、bwrap 参数级、probe 降级与工厂。

本机无 bwrap（容器环境无法创建 user namespace），bubblewrap 档
只做命令拼装与探测逻辑的参数级验证；restrictions 档全部真跑。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from mcc_config import SandboxConfig
from mcc_core import ToolContext, ToolOutput
from mcc_tools.run_bash import RunBashTool
from mcc_tools.sandbox import (
    SANDBOX_HOME_DIRNAME,
    BubblewrapExecutor,
    DirectExecutor,
    ExecTimeout,
    RestrictionsExecutor,
    build_executor,
    probe_bubblewrap,
)


async def _run(ex, command: str, cwd: Path, timeout: float = 10):
    return await ex.run(command, cwd, timeout)


# ─── DirectExecutor（零回归基线）─────────────────────────────────────────


async def test_direct_executor_runs_plain(tmp_path: Path) -> None:
    result = await _run(DirectExecutor(), "echo hello", tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == b"hello"


async def test_direct_executor_sees_full_env(tmp_path: Path,
                                              monkeypatch) -> None:
    monkeypatch.setenv("MCC_SANDBOX_TEST_SECRET", "leak-me")
    result = await _run(DirectExecutor(), "echo $MCC_SANDBOX_TEST_SECRET", tmp_path)
    assert b"leak-me" in result.stdout


async def test_executor_timeout_kills_process(tmp_path: Path) -> None:
    with pytest.raises(ExecTimeout):
        await _run(DirectExecutor(), "sleep 60", tmp_path, timeout=0.3)


# ─── RestrictionsExecutor（真执行）───────────────────────────────────────


@pytest.fixture
def restricted() -> RestrictionsExecutor:
    return RestrictionsExecutor(
        SandboxConfig(mode="restrictions", max_memory_mb=256,
                      max_cpu_secs=30, max_processes=64, max_file_mb=8)
    )


async def test_restrictions_env_whitelist_hides_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MCC_SANDBOX_TEST_SECRET", "leak-me")
    monkeypatch.setenv("TERM", "xterm")  # 白名单内应可见
    ex = RestrictionsExecutor(SandboxConfig(mode="restrictions"))

    secret = await _run(ex, "echo [$MCC_SANDBOX_TEST_SECRET]", tmp_path)
    assert b"[]" in secret.stdout  # 白名单外被剥离

    term = await _run(ex, "echo $TERM", tmp_path)
    assert b"xterm" in term.stdout


async def test_restrictions_home_redirected(tmp_path: Path) -> None:
    ex = RestrictionsExecutor(SandboxConfig(mode="restrictions"))
    result = await _run(ex, "echo $HOME", tmp_path)
    assert str(tmp_path / SANDBOX_HOME_DIRNAME) in result.stdout.decode()
    assert (tmp_path / SANDBOX_HOME_DIRNAME).is_dir()


async def test_restrictions_memory_limit_enforced(
    tmp_path: Path, restricted: RestrictionsExecutor
) -> None:
    """申请 512MB（超 256MB 上限）分配失败，命令非零退出。"""
    result = await _run(
        restricted,
        "python3 -c 'a = bytearray(512*1024*1024)' 2>&1; echo rc=$?",
        tmp_path,
    )
    text = result.stdout.decode()
    assert "MemoryError" in text or "rc=1" in text or result.returncode != 0


async def test_restrictions_file_size_limit_enforced(
    tmp_path: Path, restricted: RestrictionsExecutor
) -> None:
    """写超限文件（>8MB）被 SIGXFSZ 拦截，python 报 errno 27。"""
    await _run(
        restricted,
        "python3 -c 'open(\"big.bin\",\"wb\").write(b\"x\"*(16*1024*1024))' "
        "2>&1; echo rc=$?",
        tmp_path,
    )
    assert (tmp_path / "big.bin").stat().st_size < 16 * 1024 * 1024


async def test_restrictions_ulimits_visible(
    tmp_path: Path, restricted: RestrictionsExecutor
) -> None:
    """rlimit 在子进程内可读出（ulimit -v / -f 单位与配置换算一致）。"""
    v = await _run(restricted, "ulimit -v", tmp_path)
    assert int(v.stdout.strip()) == 256 * 1024  # KB
    f = await _run(restricted, "ulimit -f", tmp_path)
    assert int(f.stdout.strip()) == 8 * 1024  # 512B 块
    u = await _run(restricted, "ulimit -u", tmp_path)
    assert int(u.stdout.strip()) == 64


async def test_restrictions_umask_applied(tmp_path: Path) -> None:
    ex = RestrictionsExecutor(SandboxConfig(mode="restrictions"))
    await _run(ex, "touch made-in-sandbox.txt", tmp_path)
    mode = (tmp_path / "made-in-sandbox.txt").stat().st_mode & 0o777
    assert mode == 0o600


async def test_restrictions_normal_command_still_works(
    tmp_path: Path, restricted: RestrictionsExecutor
) -> None:
    result = await _run(restricted, "python3 -c 'print(40+2)'", tmp_path)
    assert result.returncode == 0
    assert b"42" in result.stdout


# ─── BubblewrapExecutor（参数级）─────────────────────────────────────────


def test_bwrap_argv_contains_isolation_bindings(tmp_path: Path) -> None:
    ex = BubblewrapExecutor(SandboxConfig(mode="bubblewrap", network="off"))
    argv = ex.build_argv("ls", tmp_path)
    joined = " ".join(argv)

    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert "--die-with-parent" in argv
    assert "--tmpfs /tmp" in joined
    assert "--ro-bind /usr /usr" in joined
    assert "--ro-bind /etc /etc" in joined
    assert f"--chdir {tmp_path}" in joined
    assert argv[-2:] == ["bash", "-c"] or argv[-1] == "ls"


def test_bwrap_argv_network_on_omits_unshare_net(tmp_path: Path) -> None:
    ex = BubblewrapExecutor(SandboxConfig(mode="bubblewrap", network="on"))
    assert "--unshare-net" not in ex.build_argv("ls", tmp_path)


# ─── probe 与工厂 ────────────────────────────────────────────────────────


def test_probe_bubblewrap_absent_binary(monkeypatch) -> None:
    import mcc_tools.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _: None)
    assert probe_bubblewrap() is False


def test_probe_bubblewrap_failing_sandbox(monkeypatch, tmp_path: Path) -> None:
    """bwrap 存在但起不了沙箱（如容器内无 user namespace）→ False。"""
    import mcc_tools.sandbox as sb

    fake = tmp_path / "bwrap"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(sb.shutil, "which", lambda _: str(fake))
    assert probe_bubblewrap() is False


def test_build_executor_mode_none_returns_direct() -> None:
    assert isinstance(build_executor(SandboxConfig()), DirectExecutor)


def test_build_executor_restrictions_returns_restrictions() -> None:
    ex = build_executor(SandboxConfig(mode="restrictions"))
    assert isinstance(ex, RestrictionsExecutor)


def test_build_executor_explicit_bwrap_unavailable_raises(monkeypatch) -> None:
    """显式点名 bubblewrap 却不可用：报错指路，绝不静默降级。"""
    import mcc_tools.sandbox as sb

    monkeypatch.setattr(sb, "probe_bubblewrap", lambda: False)
    with pytest.raises(RuntimeError, match="bwrap 不可用") as excinfo:
        build_executor(SandboxConfig(mode="bubblewrap"))
    assert "restrictions / auto" in str(excinfo.value)


def test_build_executor_auto_degrades_with_notice(monkeypatch, capsys) -> None:
    """auto 档：bwrap 不可用 → 降级 restrictions 并 stderr 提示一次。"""
    import mcc_tools.sandbox as sb

    monkeypatch.setattr(sb, "probe_bubblewrap", lambda: False)
    ex = build_executor(SandboxConfig(mode="auto"))
    assert isinstance(ex, RestrictionsExecutor)
    err = capsys.readouterr().err
    assert "降级" in err


def test_build_executor_auto_prefers_bwrap(monkeypatch) -> None:
    import mcc_tools.sandbox as sb

    monkeypatch.setattr(sb, "probe_bubblewrap", lambda: True)
    ex = build_executor(SandboxConfig(mode="auto"))
    assert isinstance(ex, BubblewrapExecutor)


# ─── run_bash 工具注入 ──────────────────────────────────────────────────


async def test_run_bash_default_executor_zero_regression(tmp_path: Path) -> None:
    tool = RunBashTool()
    ctx = ToolContext(cwd=tmp_path, session_id="s1")
    out = await tool.execute({"command": "echo direct"}, ctx)
    assert isinstance(out, ToolOutput)
    assert "direct" in out.content
    assert not out.is_error


async def test_run_bash_with_restricted_executor(tmp_path: Path,
                                                 monkeypatch) -> None:
    monkeypatch.setenv("MCC_SANDBOX_TEST_SECRET", "leak")
    tool = RunBashTool(
        RestrictionsExecutor(SandboxConfig(mode="restrictions"))
    )
    ctx = ToolContext(cwd=tmp_path, session_id="s2")
    out = await tool.execute(
        {"command": "echo [$MCC_SANDBOX_TEST_SECRET]"}, ctx
    )
    assert "[]" in out.content


async def test_run_bash_timeout_message_kept(tmp_path: Path) -> None:
    tool = RunBashTool()
    ctx = ToolContext(cwd=tmp_path, session_id="s3")
    out = await tool.execute({"command": "sleep 60", "timeout_secs": 1}, ctx)
    assert out.is_error
    assert "timed out after 1s" in out.content
