"""CLI 入口测试：version / config / headless 模式。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from click.testing import CliRunner
from mcc_cli import main
from mcc_cli.events import EchoEventSource
from mcc_cli.headless import run_headless


def test_version():
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "mcc" in result.output
    assert "0.1.0" in result.output


def test_config_default(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["config", "--cwd", str(tmp_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["model"]["main"] == "claude-opus-4-8"
    assert data["budget"]["max_usd_per_session"] == 2.0


def test_headless_outputs_response(capsys):
    source = EchoEventSource()
    rc = asyncio.run(run_headless(source, "你好", quiet=False))
    captured = capsys.readouterr()
    assert rc == 0
    assert "模拟响应" in captured.out


def test_headless_quiet_suppresses_stats(capsys):
    source = EchoEventSource()
    rc = asyncio.run(run_headless(source, "test", quiet=True))
    captured = capsys.readouterr()
    assert rc == 0
    assert "模拟响应" in captured.out
    assert "USD" not in captured.err


def test_headless_stderr_shows_tool_progress(capsys):
    source = EchoEventSource()
    asyncio.run(run_headless(source, "list files", quiet=False))
    captured = capsys.readouterr()
    assert "⚙" in captured.err
