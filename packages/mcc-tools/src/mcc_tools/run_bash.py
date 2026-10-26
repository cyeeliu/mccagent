"""run_bash 工具：执行 shell 命令，捕获 stdout/stderr，支持超时。

命令的实际隔离由注入的 SandboxExecutor 决定（默认 DirectExecutor，
行为与历史版本一致）；沙箱配置见 settings.json 的 sandbox 段。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcc_core import ToolContext, ToolOutput

from ._common import cap_str
from .sandbox import DirectExecutor, ExecResult, ExecTimeout, SandboxExecutor

if TYPE_CHECKING:
    from mcc_harness.background import BackgroundTaskManager

_CAP = 32 * 1024


class RunBashTool:
    """在会话工作目录执行 shell 命令，带超时隔离与输出截断。

    executor 可注入（沙箱档位）；缺省直跑不隔离。
    background_manager 注入后，run_in_background=True 时异步执行，
    立即返回 task_id，结果在后续轮次注入。
    """

    def __init__(
        self,
        executor: SandboxExecutor | None = None,
        background_manager: BackgroundTaskManager | None = None,
    ) -> None:
        self._executor = executor or DirectExecutor()
        self._background = background_manager

    @property
    def name(self) -> str:
        return "run_bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. stdout and stderr are captured and returned. "
            "Commands run in the session's working directory. "
            "Prefer specific tools (read_file, edit_file, grep) over shell for file operations. "
            "Set run_in_background=true for long-running commands; "
            "returns a task_id and delivers the result on a later turn."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (passed to bash -c)",
                },
                "timeout_secs": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30,
                    "description": "Seconds before the command is killed",
                },
                "run_in_background": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run asynchronously; returns task_id immediately",
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        command = input.get("command")
        if not isinstance(command, str):
            return ToolOutput.err("missing or invalid required field: command")
        timeout = input.get("timeout_secs", 30)

        if input.get("run_in_background") is True and self._background is not None:
            task_id = await self._background.start(
                command, Path(str(ctx.cwd)), float(timeout)
            )
            return ToolOutput.ok(
                f"[Background task {task_id} started. "
                "The result will be collected on a later turn.]"
            )

        try:
            result = await self._executor.run(
                command, Path(str(ctx.cwd)), float(timeout)
            )
        except ExecTimeout:
            return ToolOutput.err(f"command timed out after {timeout}s: {command}")

        return self._format(result)

    @staticmethod
    def _format(result: ExecResult) -> ToolOutput:
        stdout_str = cap_str(result.stdout.decode("utf-8", errors="replace"), _CAP)
        stderr_str = cap_str(result.stderr.decode("utf-8", errors="replace"), _CAP)

        body = ""
        if stdout_str:
            body = stdout_str
        if stderr_str:
            if body:
                body += "\n"
            body += "--- stderr ---\n"
            body += stderr_str
        if not body:
            body = "(no output)"

        if result.returncode == 0:
            return ToolOutput.ok(body)
        return ToolOutput.err(f"exit {result.returncode}\n{body}")
