"""轨迹存储测试：追加、分页、损坏行容错与后端选择。"""

import importlib.util
import json

import pytest
from eval_server.store import LocalTraceStore, TracePage, trace_store_from_env


def _lines(*seqs: int) -> list[str]:
    return [
        json.dumps({"schema_version": 1, "seq": s, "type": "llm_chunk", "turn": 0, "delta": "x"})
        for s in seqs
    ]


async def test_append_creates_jsonl_file(trace_store: LocalTraceStore, tmp_path) -> None:
    await trace_store.append("run-1", _lines(0, 1, 2))
    path = tmp_path / "traces" / "run-1.jsonl"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert len(content.splitlines()) == 3
    assert content.endswith("\n")


async def test_append_empty_list_is_noop(trace_store: LocalTraceStore) -> None:
    await trace_store.append("run-1", [])
    page = await trace_store.read_page("run-1", 0, 10)
    assert page.events == []
    assert page.total == 0


async def test_read_page_pagination(trace_store: LocalTraceStore) -> None:
    await trace_store.append("run-1", _lines(*range(10)))

    page1: TracePage = await trace_store.read_page("run-1", 0, 4)
    assert [e["seq"] for e in page1.events] == [0, 1, 2, 3]
    assert page1.total == 10
    assert page1.next_offset == 4

    page2 = await trace_store.read_page("run-1", 8, 4)
    assert [e["seq"] for e in page2.events] == [8, 9]
    assert page2.next_offset is None

    beyond = await trace_store.read_page("run-1", 20, 4)
    assert beyond.events == []
    assert beyond.next_offset is None


async def test_read_page_skips_corrupt_lines(trace_store: LocalTraceStore, tmp_path) -> None:
    path = tmp_path / "traces" / "run-1.jsonl"
    valid = _lines(0, 1, 2)
    path.write_text(
        valid[0] + "\n{broken json\n" + valid[1] + "\n" + valid[2] + "\n",
        encoding="utf-8",
    )

    page = await trace_store.read_page("run-1", 0, 10)
    assert page.total == 4  # 损坏行计入 total（对齐 PG 行为），但不出现在 events
    assert [e["seq"] for e in page.events] == [0, 1, 2]


async def test_read_missing_run_returns_empty(trace_store: LocalTraceStore) -> None:
    page = await trace_store.read_page("nope", 0, 10)
    assert page.events == []
    assert page.total == 0
    assert page.next_offset is None


def test_object_key_and_finalize(trace_store: LocalTraceStore) -> None:
    assert trace_store.object_key("run-9") == "run-9.jsonl"
    assert trace_store.finalize("run-9") == "run-9.jsonl"


async def test_health_check_roundtrip(trace_store: LocalTraceStore) -> None:
    await trace_store.health_check()


def test_from_env_defaults_to_local(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = trace_store_from_env({})
    assert isinstance(store, LocalTraceStore)

    store_dir = trace_store_from_env({"TRACE_DIR": str(tmp_path / "custom")})
    assert isinstance(store_dir, LocalTraceStore)
    assert store_dir.directory == tmp_path / "custom"


def test_from_env_s3_requires_boto3() -> None:
    if importlib.util.find_spec("boto3") is not None:
        pytest.skip("boto3 installed; ImportError path not reachable")

    with pytest.raises(ImportError, match="s3"):
        trace_store_from_env(
            {
                "TRACE_BACKEND": "s3",
                "S3_ACCESS_KEY": "k",
                "S3_SECRET_KEY": "s",
            }
        )
