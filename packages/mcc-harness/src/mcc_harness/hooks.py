"""Hook 机制：pre-tool / post-tool / lifecycle 钩子。

支持注册 Python 回调与外部 shell 脚本作为钩子，
在工具执行前后与 Agent 生命周期事件上触发。
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class HookType(Enum):
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    ON_TURN_START = "on_turn_start"
    ON_TURN_END = "on_turn_end"
    ON_ERROR = "on_error"


@dataclass
class HookContext:
    """钩子执行上下文。"""

    hook_type: HookType
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    is_error: bool = False
    session_id: str = ""
    cost_usd: float = 0.0
    iteration: int = 0
    cwd: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class HookResult:
    """钩子执行结果。

    pre-tool 钩子可设 allow=False 阻止工具执行，
    或通过 modified_input 替换工具输入。
    post-tool 钩子可通过 modified_output 替换工具输出。
    """

    allow: bool = True
    modified_input: dict[str, Any] | None = None
    modified_output: str | None = None
    message: str = ""


@runtime_checkable
class Hook(Protocol):
    """钩子协议。"""

    hook_type: HookType

    async def run(self, ctx: HookContext) -> HookResult: ...


class PythonHook:
    """Python 回调钩子。"""

    def __init__(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], Awaitable[HookResult]],
    ) -> None:
        self.hook_type = hook_type
        self._callback = callback

    async def run(self, ctx: HookContext) -> HookResult:
        return await self._callback(ctx)


class ShellHook:
    """外部 shell 脚本钩子。

    通过环境变量传递上下文，脚本退出码决定 allow/deny：
    - exit 0 → allow
    - exit non-zero → deny（stderr 作为 message）
    """

    def __init__(
        self,
        hook_type: HookType,
        script_path: str | Path,
        timeout: float = 10.0,
    ) -> None:
        self.hook_type = hook_type
        self._script = str(script_path)
        self._timeout = timeout

    async def run(self, ctx: HookContext) -> HookResult:
        env = {
            **os.environ,
            "MCC_HOOK_TYPE": ctx.hook_type.value,
            "MCC_TOOL_NAME": ctx.tool_name,
            "MCC_TOOL_INPUT": json.dumps(ctx.tool_input),
            "MCC_TOOL_OUTPUT": ctx.tool_output,
            "MCC_IS_ERROR": str(ctx.is_error),
            "MCC_SESSION_ID": ctx.session_id,
            "MCC_COST_USD": str(ctx.cost_usd),
            "MCC_ITERATION": str(ctx.iteration),
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                self._script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(ctx.cwd),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
        except (TimeoutError, FileNotFoundError, OSError) as e:
            return HookResult(allow=True, message=f"hook error: {e}")

        if proc.returncode == 0:
            return HookResult(allow=True, message=stdout.decode().strip())
        return HookResult(
            allow=False,
            message=stderr.decode().strip()
            or f"hook exited with {proc.returncode}",
        )


class HookRegistry:
    """钩子注册表：按 HookType 分组管理多个钩子。"""

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[Hook]] = defaultdict(list)

    def register(self, hook: Hook) -> None:
        self._hooks[hook.hook_type].append(hook)

    def add_python(
        self,
        hook_type: HookType,
        callback: Callable[[HookContext], Awaitable[HookResult]],
    ) -> None:
        self.register(PythonHook(hook_type, callback))

    def add_shell(
        self,
        hook_type: HookType,
        script_path: str | Path,
        timeout: float = 10.0,
    ) -> None:
        self.register(ShellHook(hook_type, script_path, timeout))

    def get(self, hook_type: HookType) -> list[Hook]:
        return self._hooks.get(hook_type, [])

    async def run_hooks(
        self, hook_type: HookType, ctx: HookContext
    ) -> list[HookResult]:
        results: list[HookResult] = []
        for hook in self.get(hook_type):
            try:
                results.append(await hook.run(ctx))
            except Exception as e:
                results.append(HookResult(allow=True, message=f"hook error: {e}"))
        return results
