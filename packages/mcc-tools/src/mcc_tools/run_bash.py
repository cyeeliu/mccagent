"""run_bash 工具：执行 shell 命令，捕获 stdout/stderr，支持超时。"""

from __future__ import annotations

import asyncio
from typing import Any

from mcc_core import ToolContext, ToolOutput

from ._common import cap_str

_CAP = 32 * 1024


class RunBashTool:
    """在会话工作目录执行 shell 命令，带超时隔离与输出截断。"""

    @property
    def name(self) -> str:
        return "run_bash"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command. stdout and stderr are captured and returned. "
            "Commands run in the session's working directory. "
            "Prefer specific tools (read_file, edit_file, grep) over shell for file operations."
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
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        command = input.get("command")
        if not isinstance(command, str):
            return ToolOutput.err("missing or invalid required field: command")
        timeout = input.get("timeout_secs", 30)

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=str(ctx.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return ToolOutput.err(f"spawn bash: {e}")

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolOutput.err(f"command timed out after {timeout}s: {command}")

        stdout_str = cap_str(stdout.decode("utf-8", errors="replace"), _CAP)
        stderr_str = cap_str(stderr.decode("utf-8", errors="replace"), _CAP)

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

        if proc.returncode == 0:
            return ToolOutput.ok(body)
        return ToolOutput.err(f"exit {proc.returncode}\n{body}")
