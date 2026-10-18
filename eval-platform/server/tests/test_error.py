"""错误类型到 HTTP 响应映射测试。"""

import json

import pytest
from eval_server.error import (
    ApiError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    UnprocessableError,
)


@pytest.mark.parametrize(
    ("err", "status_code", "code"),
    [
        (BadRequestError("bad input"), 400, "bad_request"),
        (UnauthorizedError(), 401, "unauthorized"),
        (NotFoundError("no such batch"), 404, "not_found"),
        (ConflictError("duplicate idempotency key"), 409, "conflict"),
        (UnprocessableError(), 422, "unprocessable"),
        (ApiError(), 500, "internal_error"),
    ],
)
def test_error_to_response(err: ApiError, status_code: int, code: str) -> None:
    resp = err.to_response()
    assert resp.status_code == status_code
    body = json.loads(resp.body)
    assert body["error"]["code"] == code
    assert body["error"]["message"]


def test_message_defaults_to_code() -> None:
    err = NotFoundError()
    assert err.message == "not_found"
    assert str(err) == "not_found"


def test_details_included_when_present() -> None:
    resp = NotFoundError("run not found", details={"run_id": "r-42"}).to_response()
    body = json.loads(resp.body)
    assert body["error"]["details"] == {"run_id": "r-42"}


def test_details_omitted_when_absent() -> None:
    resp = NotFoundError("run not found").to_response()
    body = json.loads(resp.body)
    assert "details" not in body["error"]


def test_errors_are_exceptions() -> None:
    with pytest.raises(ApiError):
        raise ConflictError("boom")
