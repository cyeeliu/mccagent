"""数据集测试：转换器、分页拉取与 Case 文件 IO。"""

from __future__ import annotations

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
