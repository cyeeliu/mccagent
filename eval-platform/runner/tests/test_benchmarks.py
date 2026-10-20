"""benchmarks 模块测试：转换器 / 批次构建 / 分页拉取 / 提交。"""

from __future__ import annotations

import json

import httpx
import pytest
from runner.benchmarks import (
    CODE_GEN_EXPECTATIONS,
    DATASETS,
    build_batch,
    fetch_rows,
    humaneval_case,
    mbpp_case,
    submit_batch,
    swebench_case,
)

HUMANEVAL_ROW = {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\n\ndef has_close_elements("
    "numbers: List[float], threshold: float) -> bool:\n"
    '    """ 检查任意两个数距离是否小于 threshold """\n',
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0], 0.3) == False\n",
    "entry_point": "has_close_elements",
}

MBPP_ROW = {
    "task_id": 12,
    "text": "编写一个函数查找两个数字中的最大者。",
    "test_list": ["assert maximum(1, 2) == 2", "assert maximum(3, 3) == 3"],
}

SWEBENCH_ROW = {
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "problem_statement": "修复 ascii 表读取器对非法输入的处理。",
    "FAIL_TO_PASS": '["astropy/io/ascii/tests/test_misc.py::test_read_rdb_bad_data"]',
}


def _total_weight(expectations: list[dict]) -> float:
    return sum(float(e["weight"]) for e in expectations)


class TestHumanevalCase:
    def test_converts_full_record(self):
        case = humaneval_case(HUMANEVAL_ROW)
        assert case is not None
        assert case["case_id"] == "HumanEval_0"
        for fragment in ("has_close_elements", "solution.py", "PASSED", "check(candidate)"):
            assert fragment in case["task"]
        assert case["expectations"] == CODE_GEN_EXPECTATIONS
        assert _total_weight(case["expectations"]) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "missing", ["task_id", "prompt", "test", "entry_point"], ids=lambda m: f"no_{m}"
    )
    def test_returns_none_on_missing_field(self, missing):
        row = {k: v for k, v in HUMANEVAL_ROW.items() if k != missing}
        assert humaneval_case(row) is None


class TestMbppCase:
    def test_converts_full_record(self):
        case = mbpp_case(MBPP_ROW)
        assert case is not None
        assert case["case_id"] == "mbpp_12"
        for fragment in ("solution.py", "PASSED", "maximum(1, 2) == 2"):
            assert fragment in case["task"]
        assert _total_weight(case["expectations"]) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param({"task_id": 1, "text": "x"}, id="no_tests"),
            pytest.param({"task_id": 1, "test_list": ["assert True"]}, id="no_text"),
            pytest.param({"text": "x", "test_list": ["assert True"]}, id="no_task_id"),
        ],
    )
    def test_returns_none_on_incomplete_record(self, row):
        assert mbpp_case(row) is None


class TestSwebenchCase:
    @pytest.mark.parametrize(
        ("fail_to_pass", "expected_in_task"),
        [
            pytest.param(
                '["tests/test_a.py::test_one", "tests/test_a.py::test_two"]',
                "tests/test_a.py::test_two",
                id="json_string",
            ),
            pytest.param(
                ["tests/test_b.py::test_x"], "tests/test_b.py::test_x", id="native_list"
            ),
            pytest.param(None, "未提供", id="missing"),
        ],
    )
    def test_converts_record(self, fail_to_pass, expected_in_task):
        row = dict(SWEBENCH_ROW, FAIL_TO_PASS=fail_to_pass)
        case = swebench_case(row)
        assert case is not None
        assert case["case_id"] == "astropy__astropy-12907"
        assert "git clone https://github.com/astropy/astropy.git" in case["task"]
        assert "DONE" in case["task"]
        assert expected_in_task in case["task"]
        assert _total_weight(case["expectations"]) == pytest.approx(1.0)

    def test_returns_none_on_missing_problem(self):
        row = {k: v for k, v in SWEBENCH_ROW.items() if k != "problem_statement"}
        assert swebench_case(row) is None


class TestBuildBatch:
    def test_skips_incomplete_records(self, capsys):
        batch = build_batch(
            "humaneval",
            [HUMANEVAL_ROW, {"task_id": "HumanEval/1"}],
            name="demo",
        )
        assert [c["case_id"] for c in batch["cases"]] == ["HumanEval_0"]
        assert batch["profile"] == "mcc-main"
        assert "skipped 1" in capsys.readouterr().err

    def test_raises_when_no_valid_case(self):
        with pytest.raises(ValueError, match="no valid cases"):
            build_batch("mbpp", [{"task_id": 1}])

    def test_default_batch_name_contains_dataset(self):
        batch = build_batch("humaneval", [HUMANEVAL_ROW])
        assert batch["name"] == "humaneval-1"


def _rows_api_handler(
    pages: list[list[dict]], requests: list[httpx.Request]
) -> httpx.MockTransport:
    """构造 datasets-server rows API 的分页 mock：第 N 次请求返回第 N 页。"""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        offset = int(dict(request.url.params)["offset"])
        page = pages[len(requests) - 1]
        return httpx.Response(
            200,
            json={
                "rows": [{"row_idx": offset + i, "row": row} for i, row in enumerate(page)],
                "num_rows_total": sum(len(p) for p in pages),
            },
        )

    return httpx.MockTransport(handler)


class TestFetchRows:
    def test_paginates_and_merges(self):
        requests: list[httpx.Request] = []
        transport = _rows_api_handler(
            [[HUMANEVAL_ROW, dict(HUMANEVAL_ROW, task_id="HumanEval/1")],
             [dict(HUMANEVAL_ROW, task_id="HumanEval/2")]],
            requests,
        )
        with httpx.Client(transport=transport) as client:
            rows = fetch_rows(DATASETS["humaneval"], limit=0, client=client)
        assert [r["task_id"] for r in rows] == ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
        assert len(requests) == 2

    def test_limit_truncates_after_first_page(self):
        requests: list[httpx.Request] = []
        transport = _rows_api_handler(
            [[HUMANEVAL_ROW, dict(HUMANEVAL_ROW, task_id="HumanEval/1")]],
            requests,
        )
        with httpx.Client(transport=transport) as client:
            rows = fetch_rows(DATASETS["humaneval"], limit=1, client=client)
        assert len(rows) == 1
        assert len(requests) == 1


class TestSubmitBatch:
    def _make_server(self, profile_exists: bool) -> tuple[httpx.MockTransport, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/api/profiles" and request.method == "GET":
                items = [{"name": "mcc-main"}] if profile_exists else []
                return httpx.Response(200, json={"items": items})
            if request.url.path == "/api/profiles":
                return httpx.Response(200, json={"id": "p1"})
            if request.url.path == "/api/batches":
                return httpx.Response(200, json={"id": "b1"})
            raise AssertionError(f"unexpected request: {request}")

        return httpx.MockTransport(handler), requests

    def test_creates_profile_then_submits(self):
        batch = build_batch("humaneval", [HUMANEVAL_ROW])
        transport, requests = self._make_server(profile_exists=False)
        with httpx.Client(transport=transport) as client:
            result = submit_batch(batch, server="http://srv:8081", client=client)
        assert result == {"id": "b1"}
        methods_paths = [(r.method, str(r.url.path)) for r in requests]
        assert ("GET", "/api/profiles") in methods_paths
        assert ("POST", "/api/profiles") in methods_paths
        assert methods_paths[-1] == ("POST", "/api/batches")
        profile_body = json.loads(requests[1].content)
        assert profile_body["scaffold"] == "mini-claude-code"

    def test_reuses_existing_profile(self):
        batch = build_batch("humaneval", [HUMANEVAL_ROW])
        transport, requests = self._make_server(profile_exists=True)
        with httpx.Client(transport=transport) as client:
            submit_batch(batch, server="http://srv:8081", client=client)
        assert all(
            not (r.method == "POST" and str(r.url.path) == "/api/profiles") for r in requests
        )
