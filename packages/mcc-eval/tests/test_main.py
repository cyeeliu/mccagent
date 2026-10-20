"""CLI 测试：子命令端到端与模型解析优先级。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from mcc_eval.__main__ import _non_negative_int, build_parser, main, resolve_model

EXAMPLES = Path(__file__).parents[1] / "examples" / "mock-cases.json"


def test_run_mock_end_to_end(tmp_path: Path, capsys) -> None:
    """run --mock 全链路：产物四件套齐备，汇总正确。"""
    out = tmp_path / "runs"
    rc = main(["run", "--cases", str(EXAMPLES), "--mock", "--out", str(out)])
    assert rc == 0

    batch_dirs = list(out.iterdir())
    assert len(batch_dirs) == 1
    batch_dir = batch_dirs[0]
    assert (batch_dir / "cases.json").exists()
    assert (batch_dir / "report.md").exists()
    assert (batch_dir / "runs" / "demo-pass.jsonl").exists()

    data = json.loads((batch_dir / "results.json").read_text(encoding="utf-8"))
    assert data["model"] == "mock"
    assert data["summary"]["total"] == 2
    assert data["summary"]["passed"] == 1
    by_id = {r["case_id"]: r["status"] for r in data["results"]}
    assert by_id == {"demo-pass": "passed", "demo-fail": "failed"}
    assert "通过率" in capsys.readouterr().err


def test_report_subcommand_regrades(tmp_path: Path, capsys) -> None:
    out = tmp_path / "runs"
    main(["run", "--cases", str(EXAMPLES), "--mock", "--out", str(out)])
    batch_dir = next(out.iterdir())

    # 篡改 results.json 的结论，report 重评分后应恢复
    results_path = batch_dir / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    data["results"][0]["status"] = "error"
    results_path.write_text(json.dumps(data), encoding="utf-8")

    capsys.readouterr()
    rc = main(["report", str(batch_dir)])
    assert rc == 0
    restored = json.loads(results_path.read_text(encoding="utf-8"))
    assert restored["results"][0]["status"] == "passed"
    assert "重评完成" in capsys.readouterr().err


def test_datasets_listing(capsys) -> None:
    rc = main(["datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    for key in ("humaneval", "mbpp", "swebench-lite"):
        assert key in out


def test_report_missing_dir_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="批次目录不存在"):
        main(["report", str(tmp_path / "nope")])


def test_run_unknown_dataset_exits() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--dataset", "bogus"])


def test_parser_requires_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run"])


@pytest.mark.parametrize("bad", ["-1", "-5"])
def test_non_negative_int_rejects(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=">= 0"):
        _non_negative_int(bad)


@pytest.mark.parametrize("good", ["0", "10"])
def test_non_negative_int_accepts(good: str) -> None:
    assert _non_negative_int(good) == int(good)


# ─── 模型解析优先级 ──────────────────────────────────────────────────────


def test_resolve_model_explicit_wins(monkeypatch) -> None:
    monkeypatch.setenv("MODEL", "from-env")
    assert resolve_model("explicit") == "explicit"


def test_resolve_model_env_beats_settings(monkeypatch) -> None:
    monkeypatch.setenv("MODEL", "from-env")

    class Settings:
        class model:
            main = "from-settings"

    assert resolve_model(None, settings_loader=lambda: Settings) == "from-env"


def test_resolve_model_settings_beats_default() -> None:
    class Settings:
        class model:
            main = "from-settings"

    assert resolve_model(None, settings_loader=lambda: Settings) == "from-settings"


def test_resolve_model_defaults_when_all_missing(monkeypatch) -> None:
    monkeypatch.delenv("MODEL", raising=False)

    def broken_loader():
        raise RuntimeError("no mcc config")

    assert resolve_model(None, settings_loader=broken_loader) == "claude-sonnet-4-5"
