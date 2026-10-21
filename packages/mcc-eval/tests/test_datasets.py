"""数据集测试：转换器、分页拉取与 Case 文件 IO。"""

from __future__ import annotations

import argparse
import json

import httpx
import pytest
from mcc_eval.datasets import (
    DATASETS,
    Case,
    build_cases,
    export_cases_file,
    fetch_rows,
    humaneval_case,
    load_cases_file,
    mbpp_case,
    swebench_case,
)

HUMANEVAL_ROW = {
    "task_id": "HumanEval/0",
    "prompt": "def return_hello():\n    \"\"\"say hello\"\"\"\n",
    "test": "def check(candidate):\n    assert candidate() == 'hello'\n",
    "entry_point": "return_hello",
}

MBPP_ROW = {
    "task_id": 2,
    "text": "Write a function to find the minimum of two numbers.",
    "test_list": ["assert min_num(2,3)==2", "assert min_num(10,5)==5"],
}

SWEBENCH_ROW = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "problem_statement": "Modeling's separability_matrix does not compute…",
    "FAIL_TO_PASS": '["astropy/modeling/tests/test_core.py::test_separable"]',
}


# ─── 转换器 ───────────────────────────────────────────────────────────────


def test_humaneval_case_bundles_official_test() -> None:
    """HumanEval 转换保留官方测试代码（执行式评分的判定依据）。"""
    case = humaneval_case(HUMANEVAL_ROW)
    assert case is not None
    assert case.case_id == "HumanEval_0"
    assert "solution.py" in case.task
    assert "from solution import return_hello as candidate" in (case.check or "")
    assert "check(candidate)" in (case.check or "")
    assert case.expectations  # 断言明细保留作诊断


def test_mbpp_case_joins_assertions() -> None:
    case = mbpp_case(MBPP_ROW)
    assert case is not None
    assert case.case_id == "mbpp_2"
    assert "from solution import *" in (case.check or "")
    assert "assert min_num(2,3)==2" in (case.check or "")
    assert "assert min_num(10,5)==5" in (case.check or "")


def test_swebench_case_uses_behavioral_assertions() -> None:
    """SWE-bench 无官方测试（需 Docker 任务容器），走行为断言。"""
    case = swebench_case(SWEBENCH_ROW)
    assert case is not None
    assert case.check is None
    assert case.case_id == "astropy__astropy-12907"
    assert "git clone https://github.com/astropy/astropy.git" in case.task
    types = {e["type"] for e in case.expectations}
    assert "tool_called" in types and "output_contains" in types


@pytest.mark.parametrize(
    "converter",
    [humaneval_case, mbpp_case, swebench_case],
    ids=lambda fn: fn.__name__,
)
def test_converters_skip_incomplete_records(converter) -> None:
    assert converter({"task_id": "x"}) is None
    assert converter({}) is None


def test_mbpp_json_string_test_list_is_accepted() -> None:
    """HF 数据集里列表字段可能被序列化为 JSON 字符串。"""
    row = dict(MBPP_ROW, test_list='["assert f(1)==1"]')
    case = mbpp_case(row)
    assert case is not None
    assert "assert f(1)==1" in (case.check or "")


def test_swebench_fail_to_pass_json_string_is_accepted() -> None:
    case = swebench_case(SWEBENCH_ROW)
    assert case is not None
    assert "test_separable" in case.task


def test_build_cases_skips_and_collects() -> None:
    cases = build_cases("humaneval", [HUMANEVAL_ROW, {"task_id": "broken"}])
    assert [c.case_id for c in cases] == ["HumanEval_0"]


def test_build_cases_raises_when_all_invalid() -> None:
    with pytest.raises(ValueError, match="yielded no valid cases"):
        build_cases("humaneval", [{"task_id": "broken"}])


# ─── 分页拉取（MockTransport，不出网）────────────────────────────────────


def _mock_transport(pages: list[list[dict]], total: int) -> httpx.MockTransport:
    state = {"offset": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        idx = state["offset"]
        state["offset"] += 1
        assert offset == idx * 100  # 按页推进
        return httpx.Response(200, json={"rows": [{"row": r} for r in pages[idx]],
                                     "num_rows_total": total})

    return httpx.MockTransport(handler)


def test_fetch_rows_paginates_until_limit() -> None:
    pages = [[{"i": n} for n in range(100)], [{"i": n} for n in range(100, 150)]]
    client = httpx.Client(transport=_mock_transport(pages, total=150))
    with client:
        rows = fetch_rows(DATASETS["humaneval"], limit=120, client=client)
    assert len(rows) == 120
    assert rows[-1]["i"] == 119


def test_fetch_rows_full_dataset() -> None:
    pages = [[{"i": n} for n in range(100)], [{"i": n} for n in range(100, 130)]]
    client = httpx.Client(transport=_mock_transport(pages, total=130))
    with client:
        rows = fetch_rows(DATASETS["humaneval"], limit=0, client=client)
    assert len(rows) == 130


# ─── Case 文件 IO ─────────────────────────────────────────────────────────


def test_case_dict_roundtrip(tmp_path) -> None:
    cases = [humaneval_case(HUMANEVAL_ROW), swebench_case(SWEBENCH_ROW)]
    assert all(c is not None for c in cases)
    path = tmp_path / "cases.json"
    export_cases_file(cases, path)  # type: ignore[arg-type]
    loaded = load_cases_file(path)
    assert loaded == cases


def test_case_from_dict_requires_id_and_task() -> None:
    with pytest.raises(ValueError, match="case_id and task"):
        Case.from_dict({"case_id": "x"})


def test_case_setup_roundtrip() -> None:
    """setup 环境预置随 case 落盘 / 读取往返保持。"""
    case = Case(case_id="s1", task="修复 config 解析", setup={
        "config/app.yaml": "port: 8081\n",
        "tests/test_app.py": "def test_port(): ...\n",
    })
    raw = case.to_dict()
    assert raw["setup"]["config/app.yaml"] == "port: 8081\n"
    assert Case.from_dict(raw) == case


@pytest.mark.parametrize("bad", [{"a": 1}, ["x"], "file"])
def test_case_from_dict_rejects_non_mapping_setup(bad) -> None:
    with pytest.raises(ValueError, match="setup must be"):
        Case.from_dict({"case_id": "x", "task": "t", "setup": bad})


def test_load_cases_file_rejects_non_array(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"case_id": "x", "task": "t"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_cases_file(path)


def test_dataset_registry_shape() -> None:
    """三个业界基准注册齐全，spec 字段完整。"""
    assert set(DATASETS) == {"swebench-lite", "humaneval", "mbpp"}
    for spec in DATASETS.values():
        assert spec.hf_id and spec.config and spec.split and callable(spec.converter)


# ─── 网络重试与异常处理 ───────────────────────────────────────────────────


def _pages_payload(rows: list[dict], total: int) -> dict:
    return {"rows": [{"row": r} for r in rows], "num_rows_total": total}


def test_fetch_rows_retries_transient_connection_reset(monkeypatch) -> None:
    """连接重置后重试成功：瞬时传输错误不终结拉取。"""
    monkeypatch.setattr("mcc_eval.datasets.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("[Errno 104] Connection reset by peer")
        return httpx.Response(200, json=_pages_payload([{"i": 1}], total=1))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with client:
        rows = fetch_rows(DATASETS["humaneval"], limit=0, client=client)

    assert rows == [{"i": 1}]
    assert calls["n"] == 2  # 首发 + 1 次重试


def test_fetch_rows_retries_5xx_then_succeeds(monkeypatch) -> None:
    """datasets-server 偶发 502：指数退避后成功。"""
    monkeypatch.setattr("mcc_eval.datasets.time.sleep", lambda _: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(502, json={"error": "bad gateway"})
        return httpx.Response(200, json=_pages_payload([{"i": 7}], total=1))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with client:
        rows = fetch_rows(DATASETS["humaneval"], limit=0, client=client)

    assert rows == [{"i": 7}]
    assert calls["n"] == 3


def test_fetch_rows_exhausted_retries_raises_runtimeerror(monkeypatch) -> None:
    """持续连接重置：重试耗尽抛 RuntimeError（含请求与次数信息）。"""
    monkeypatch.setattr("mcc_eval.datasets.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 104] Connection reset by peer")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="已重试 3 次"), client:
        fetch_rows(DATASETS["humaneval"], limit=0, client=client)


def test_fetch_rows_does_not_retry_404() -> None:
    """数据集不存在（404）不重试：立刻失败不空转。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"error": "dataset not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError), client:
        fetch_rows(DATASETS["humaneval"], limit=0, client=client)
    assert calls["n"] == 1


def test_cli_gather_exit_message_points_to_offline_path(monkeypatch, tmp_path) -> None:
    """重试耗尽后 CLI 给出离线导出的替代指引，而非裸 traceback。"""
    from mcc_eval.__main__ import _gather_cases

    monkeypatch.setattr("mcc_eval.datasets.time.sleep", lambda _: None)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 104] Connection reset by peer")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    real_fetch = __import__("mcc_eval.datasets", fromlist=["fetch_rows"]).fetch_rows

    def broken_fetch(spec, limit=0, client=None):
        return real_fetch(spec, limit, client=client)

    monkeypatch.setattr(
        "mcc_eval.__main__.fetch_rows",
        lambda spec, limit: broken_fetch(spec, limit, client=client),
    )
    args = argparse.Namespace(cases=None, dataset="swebench-lite", limit=5)

    with pytest.raises(SystemExit) as excinfo:
        _gather_cases(args)
    msg = str(excinfo.value)
    assert "拉取 SWE-bench Lite 失败" in msg
    assert "datasets --dataset swebench-lite --limit 5 --out cases.json" in msg
    assert "run --cases cases.json" in msg
