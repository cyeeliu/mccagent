"""统一错误类型：业务异常到 HTTP 响应的映射。

所有可预期错误都抛 ApiError 子类，由异常处理器统一转成
``{"error": {"code", "message", "details"}}`` 结构的响应。
"""

from __future__ import annotations

from typing import Any

from fastapi import status
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """可预期业务错误基类。"""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details

    def to_response(self) -> JSONResponse:
        payload: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.details is not None:
            payload["error"]["details"] = self.details
        return JSONResponse(status_code=self.status_code, content=payload)


class BadRequestError(ApiError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"


class UnauthorizedError(ApiError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class NotFoundError(ApiError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ConflictError(ApiError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class UnprocessableError(ApiError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unprocessable"
