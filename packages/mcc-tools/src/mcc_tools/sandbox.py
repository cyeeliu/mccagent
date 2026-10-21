"""bash 命令沙箱：run_bash 的物理隔离执行层。

权限系统是逻辑层（「应该做」），沙箱是物理层（「能做」）——即使
prompt injection 骗过模型与权限检查，破坏性操作在本层物理不可达。
两层正交：deny 规则永远先行，沙箱在其后兜底。

四档隔离强度（见 SandboxConfig.mode）：

  none          不隔离（默认，与历史行为一致）
  restrictions  纯 Python 资源限制：rlimit（内存 / CPU / 进程数 /
                文件大小）+ 环境变量白名单 + umask 077 + HOME 重定向
                到工作目录下的 .sandbox-home。无法限制网络。
  bubblewrap    bwrap 命名空间隔离：系统目录只读绑定、独立 /tmp
                （tmpfs）、禁网（--unshare-net）、随宿主进程退出
                （--die-with-parent）。
  auto          依次探测 bubblewrap → restrictions，降级在 stderr
                提示一次。

exec接口统一为 ``SandboxExecutor.run(command, cwd, timeout)``，
返回 ``ExecResult``；超时由各执行器自行 kill 并抛 ``ExecTimeout``。
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import subprocess
import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mcc_config import SandboxConfig

SANDBOX_HOME_DIRNAME = ".sandbox-home"
PROBE_TIMEOUT_S = 5.0


@dataclass
class ExecResult:
    """一次命令执行的原始结果（截断与拼接由 run_bash 负责）。"""

    returncode: int
    stdout: bytes
    stderr: bytes


class ExecTimeout(Exception):
    """命令超时（执行器已 kill 子进程并等待回收）。"""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"command timed out after {timeout}s")
        self.timeout = timeout


class SandboxExecutor(Protocol):
    """命令执行器协议：Direct / Restrictions / Bubblewrap 共用。"""

    def run(self, command: str, cwd: Path, timeout: float) -> Awaitable[ExecResult]:
        ...


async def _run_argv(
    argv: list[str], cwd: Path, timeout: float,
    *, env: dict[str, str] | None = None,
    preexec_fn=None,
) -> ExecResult:
    """spawn → communicate → 超时 kill 的公共底座。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            preexec_fn=preexec_fn,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return ExecResult(returncode=127, stdout=b"", stderr=f"spawn: {e}".encode())
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise ExecTimeout(timeout) from None
    return ExecResult(
        returncode=proc.returncode or 0, stdout=stdout, stderr=stderr
    )


class DirectExecutor:
    """不隔离：直接 bash -c（历史行为原样迁入）。"""

    async def run(self, command: str, cwd: Path, timeout: float) -> ExecResult:
        return await _run_argv(["bash", "-c", command], cwd, timeout)


class RestrictionsExecutor:
    """纯 Python 资源限制档：rlimit + env 白名单 + umask + HOME 重定向。

    preexec_fn 只执行 setrlimit / umask 这类窄窗口操作（async-signal
    安全约束下的通行做法）；环境变量重建走 spawn 的 env= 参数，
    不进 preexec。无法限制网络——需要禁网用 bubblewrap 档。

    RLIMIT_NPROC 语义是 per-uid（同一用户的进程总数上限）：单用户
    工作站 / 容器场景即等价于限制本沙箱；共享多用户机器上请评估
    该 uid 的既有进程数。
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._cfg = config

    def _limits(self):
        """构造 preexec_fn 闭包：在子进程内施加 rlimit 与 umask。

        umask 仅在 allowlist 策略下应用（077）：inherit 是 harness
        日常档，尊重用户 shell 的 umask，避免 Agent 创建的文件
        权限与用户预期不一致。
        """
        cfg = self._cfg

        def _preexec() -> None:
            if cfg.max_memory_mb > 0:
                _set(resource.RLIMIT_AS, cfg.max_memory_mb * 1024 * 1024)
            if cfg.max_cpu_secs > 0:
                _set(resource.RLIMIT_CPU, cfg.max_cpu_secs)
            if cfg.max_processes > 0:
                _set(resource.RLIMIT_NPROC, cfg.max_processes)
            if cfg.max_file_mb > 0:
                _set(resource.RLIMIT_FSIZE, cfg.max_file_mb * 1024 * 1024)
            if cfg.env_policy == "allowlist":
                os.umask(0o077)

        def _set(res: int, limit: int) -> None:
            resource.setrlimit(res, (limit, limit))

        return _preexec

    def _build_env(self, cwd: Path) -> dict[str, str] | None:
        """按 env_policy 构建子进程环境。

        - inherit：返回 None（spawn 继承完整环境）——harness 日常
          档，Agent 能用项目 venv / cargo 等用户工具链
        - allowlist：白名单重建；PATH 一律重置为系统默认而非继承
          父进程值（否则 harness 自身的 venv PATH 会泄漏给被测
          Agent，评测环境被污染）；HOME 按 redirect_home 决定是否
          重定向到工作目录内
        """
        if self._cfg.env_policy == "inherit":
            return None
        env = {k: os.environ[k] for k in self._cfg.env_allow if k in os.environ}
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        if self._cfg.redirect_home:
            env["HOME"] = str(cwd / SANDBOX_HOME_DIRNAME)
        else:
            env.setdefault("HOME", os.environ.get("HOME", str(cwd)))
        env.setdefault("TMPDIR", "/tmp")
        return env

    async def run(self, command: str, cwd: Path, timeout: float) -> ExecResult:
        if self._cfg.env_policy == "allowlist" and self._cfg.redirect_home:
            (cwd / SANDBOX_HOME_DIRNAME).mkdir(exist_ok=True)
        return await _run_argv(
            ["bash", "-c", command],
            cwd,
            timeout,
            env=self._build_env(cwd),
            preexec_fn=self._limits(),
        )


class BubblewrapExecutor:
    """bubblewrap 档：命名空间隔离（禁网 + 系统目录只读 + 独立 /tmp）。

    绑定集：/usr /bin /lib /lib64 /etc 只读，/proc /dev 伪文件系统，
    /tmp 为空 tmpfs，工作目录可写。Agent 退出后沙箱内进程一并回收
    （--die-with-parent），不残留。
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._cfg = config

    def build_argv(self, command: str, cwd: Path) -> list[str]:
        """拼装 bwrap 命令行（独立方法便于参数级测试）。"""
        argv = ["bwrap"]
        for src in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            argv += ["--ro-bind", src, src]
        argv += [
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--die-with-parent",
            "--chdir", str(cwd),
        ]
        if self._cfg.network == "off":
            argv.append("--unshare-net")
        argv += ["bash", "-c", command]
        return argv

    async def run(self, command: str, cwd: Path, timeout: float) -> ExecResult:
        home = cwd / SANDBOX_HOME_DIRNAME
        home.mkdir(exist_ok=True)
        # bwrap 档恒定环境清洗（强隔离语义）：真实 HOME 在沙箱内
        # 不可见，HOME 必须重定向到工作目录内；PATH 重置为系统默认
        env = {k: os.environ[k] for k in self._cfg.env_allow if k in os.environ}
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["HOME"] = str(home)
        return await _run_argv(
            self.build_argv(command, cwd), cwd, timeout, env=env
        )


def probe_bubblewrap() -> bool:
    """探测 bwrap 是否可用：可执行 + 能起最小沙箱。

    部分容器环境 bwrap 存在但因缺少权能无法创建 user namespace，
    因此不能只看 which——必须真跑一次最小绑定。
    """
    if shutil.which("bwrap") is None:
        return False
    try:
        result = subprocess.run(
            [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--dev", "/dev",
                "--proc", "/proc",
                "--tmpfs", "/tmp",
                "true",
            ],
            capture_output=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def build_executor(config: SandboxConfig):
    """工厂：按配置构建执行器；auto 探测降级并提示。

    显式 bubblewrap 但环境不可用时抛 RuntimeError——用户点名要
    命名空间隔离却静默降级到弱档是安全语义错误，必须报错指路。
    """
    if config.mode == "none":
        return DirectExecutor()
    if config.mode == "restrictions":
        return RestrictionsExecutor(config)
    if config.mode == "bubblewrap":
        if not probe_bubblewrap():
            raise RuntimeError(
                "sandbox.mode=bubblewrap 但 bwrap 不可用（未安装，或当前"
                "环境缺少创建 user namespace 的权能）。安装 bubblewrap "
                "或改用 restrictions / auto。"
            )
        return BubblewrapExecutor(config)

    # auto：bubblewrap → restrictions 降级
    if probe_bubblewrap():
        return BubblewrapExecutor(config)
    print(
        "沙箱 auto 档：bwrap 不可用，降级为 restrictions（资源限制，"
        "不含网络隔离）",
        file=sys.stderr,
    )
    return RestrictionsExecutor(config)
