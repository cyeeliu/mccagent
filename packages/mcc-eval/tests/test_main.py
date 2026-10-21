"""CLI 测试：子命令端到端、多次采样、失败回流与模型解析优先级。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from mcc_eval.__main__ import (
    _non_negative_int,
    _positive_int,
    build_parser,
    main,
)

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
    assert (batch_dir / "runs" / "demo-pass-a1.jsonl").exists()

    data = json.loads((batch_dir / "results.json").read_text(encoding="utf-8"))
    assert data["model"] == "mock"
    assert data["summary"]["total"] == 2
    assert data["summary"]["passed"] == 1
    assert data["summary"]["stable_cases"] == 1
    assert data["summary"]["flaky_cases"] == 0
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


def test_run_multiple_samples_flaky_detection(tmp_path: Path, capsys) -> None:
    """--runs-per-case 2：每 case 两份轨迹，demo-fail 两次都挂（稳定失败）。"""
    out = tmp_path / "runs"
    rc = main(["run", "--cases", str(EXAMPLES), "--mock", "--out", str(out),
               "--runs-per-case", "2"])
    assert rc == 0

    batch_dir = next(out.iterdir())
    assert (batch_dir / "runs" / "demo-pass-a1.jsonl").exists()
    assert (batch_dir / "runs" / "demo-pass-a2.jsonl").exists()
    assert (batch_dir / "work" / "demo-fail-a2").is_dir()

    data = json.loads((batch_dir / "results.json").read_text(encoding="utf-8"))
    assert data["summary"]["total"] == 4
    assert data["summary"]["cases"] == 2
    assert data["summary"]["stable_cases"] == 1  # demo-pass 两次全过
    assert data["summary"]["flaky_cases"] == 0   # demo-fail 两次全挂，非 flaky
    attempts = {r["attempt"] for r in data["results"] if r["case_id"] == "demo-pass"}
    assert attempts == {1, 2}

    report = (batch_dir / "report.md").read_text(encoding="utf-8")
    assert "## Case 级聚合（k/N 次采样通过）" in report
    assert "| demo-pass | 2/2 | 稳定通过 |" in report
    assert "| demo-fail | 0/2 | 稳定失败 |" in report


@pytest.mark.parametrize("bad", ["0", "-3"])
def test_positive_int_rejects(bad: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
        _positive_int(bad)


@pytest.mark.parametrize("good", ["1", "5"])
def test_positive_int_accepts(good: str) -> None:
    assert _positive_int(good) == int(good)


def test_distill_subcommand_creates_drafts(tmp_path: Path, capsys,
                                            monkeypatch) -> None:
    """distill：失败会话回流为草稿，幂等（二跑不覆盖）。"""
    sessions = _make_session(tmp_path, sid="abc12345", status="errored",
                             task="修复 config 解析崩溃")
    _make_session(tmp_path, sid="good0001", status="ended", task="正常会话")

    out = tmp_path / "cases"
    monkeypatch.chdir(tmp_path)
    rc = main(["distill", "--sessions-dir", str(sessions),
               "--out", str(out)])
    assert rc == 0

    draft = out / "distill-abc12345.json"
    assert draft.exists()
    payload = json.loads(draft.read_text(encoding="utf-8"))
    assert payload[0]["case_id"] == "distill-abc12345"
    assert payload[0]["task"] == "修复 config 解析崩溃"
    assert payload[0]["expectations"] == []
    assert not (out / "distill-good0001.json").exists()  # ended 不回流

    err = capsys.readouterr().err
    assert "新建 1" in err
    assert "人工补齐" in err

    rc = main(["distill", "--sessions-dir", str(sessions), "--out", str(out)])
    assert rc == 0
    assert "跳过（已存在）1" in capsys.readouterr().err
    assert json.loads(draft.read_text(encoding="utf-8")) == payload  # 未被覆盖


def _make_session(base: Path, *, sid: str, status: str, task: str) -> Path:
    """构造一个 mcc 会话目录（meta + JSONL 流水）。"""
    day = base / "sessions" / "2026-10-20"
    day.mkdir(parents=True, exist_ok=True)
    meta = {"id": sid, "started_at": "2026-10-20T10:00:00+00:00",
            "cwd": "/tmp", "turns": 1, "cost_usd": 0.01, "status": status}
    (day / f"{sid}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    snapshot = {
        "ts": "2026-10-20T10:00:01+00:00", "iteration": 1,
        "request_messages": [
            {"role": "user", "content": [{"type": "text", "text": task}]},
        ],
        "assistant_blocks": [], "tool_outputs": [],
        "usage": {}, "model": "m", "cost_usd": 0.01,
    }
    (day / f"{sid}.jsonl").write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
    return base / "sessions"


def test_distill_missing_dir_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="会话目录不存在"):
        main(["distill", "--sessions-dir", str(tmp_path / "nope")])


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


# 模型解析优先级测试已随实现迁至 tests/test_config_resolve.py
# （mcc-config 是模型配置的唯一入口）
