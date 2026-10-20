"""轨迹存储：append-only JSONL，后端为本地目录或 S3 兼容对象存储。

设计原则：
- 完整轨迹永不进数据库——数据库只存索引，原始事件流存对象存储；
- 写入只追加、绝不覆盖，单条轨迹可能达数百 MB；
- 读取按事件行数分页，进行中的 run 也可读；
- 后端在本地与 S3 之间切换时接口不变。

配置：
- TRACE_BACKEND=local（默认）：本地 TRACE_DIR 目录
- TRACE_BACKEND=s3：S3 / MinIO，需 S3_* 环境变量（安装 s3 extra）
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TracePage:
    """轨迹分页结果；next_offset 为 None 表示已到末尾。"""

    events: list[dict[str, Any]]
    total: int
    next_offset: int | None


class TraceStoreProtocol(Protocol):
    """轨迹存储后端接口；本地与 S3 实现保持同一契约。"""

    async def append(self, run_id: str, lines: list[str]) -> None: ...

    async def read_page(self, run_id: str, offset: int, limit: int) -> TracePage: ...

    def finalize(self, run_id: str) -> str: ...

    def object_key(self, run_id: str) -> str: ...

    async def health_check(self) -> None: ...


def _parse_page(content: str, offset: int, limit: int) -> TracePage:
    all_lines = [ln for ln in content.splitlines() if ln.strip()]
    total = len(all_lines)
    events: list[dict[str, Any]] = []
    for line in all_lines[offset : offset + limit]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    next_offset = offset + limit if offset + limit < total else None
    return TracePage(events=events, total=total, next_offset=next_offset)


class LocalTraceStore:
    """本地文件系统后端：每个 run 一个 {run_id}.jsonl 文件。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def object_key(self, run_id: str) -> str:
        return f"{run_id}.jsonl"

    def _path(self, run_id: str) -> Path:
        return self.directory / self.object_key(run_id)

    async def append(self, run_id: str, lines: list[str]) -> None:
        if not lines:
            return
        payload = "".join(f"{line}\n" for line in lines)
        await asyncio.to_thread(self._append_sync, self._path(run_id), payload)

    @staticmethod
    def _append_sync(path: Path, payload: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())

    async def read_page(self, run_id: str, offset: int, limit: int) -> TracePage:
        path = self._path(run_id)
        content = await asyncio.to_thread(
            lambda: path.read_text(encoding="utf-8") if path.exists() else ""
        )
        return _parse_page(content, offset, limit)

    def finalize(self, run_id: str) -> str:
        return self.object_key(run_id)

    async def health_check(self) -> None:
        probe = self.directory / "_healthcheck"
        await asyncio.to_thread(probe.write_text, "ok", encoding="utf-8")
        if await asyncio.to_thread(probe.exists):
            await asyncio.to_thread(probe.unlink)


class S3TraceStore:
    """S3 / MinIO 后端。

    S3 不支持原生追加，采用「读旧 + 追加 + 写回」；高频上报会产生写放大，
    生产可改为分段 part 上传、完成时 multipart 合并，此处保持接口简单。
    依赖 boto3（uv sync --extra s3），未安装时构造抛出明确错误。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        prefix: str = "",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "boto3 not installed. Run: uv sync --extra s3"
            ) from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def object_key(self, run_id: str) -> str:
        name = f"{run_id}.jsonl"
        return f"{self.prefix}/{name}" if self.prefix else name

    async def append(self, run_id: str, lines: list[str]) -> None:
        if not lines:
            return
        key = self.object_key(run_id)
        existing = await asyncio.to_thread(self._get_object_bytes, key)
        payload = existing + "".join(f"{line}\n" for line in lines).encode("utf-8")
        await asyncio.to_thread(
            self.client.put_object, Bucket=self.bucket, Key=key, Body=payload
        )

    def _get_object_bytes(self, key: str) -> bytes:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return resp["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return b""

    async def read_page(self, run_id: str, offset: int, limit: int) -> TracePage:
        key = self.object_key(run_id)
        content = await asyncio.to_thread(self._get_object_bytes, key)
        return _parse_page(content.decode("utf-8", errors="replace"), offset, limit)

    def finalize(self, run_id: str) -> str:
        return self.object_key(run_id)

    async def health_check(self) -> None:
        key = "_healthcheck"
        await asyncio.to_thread(
            self.client.put_object, Bucket=self.bucket, Key=key, Body=b"ok"
        )
        await asyncio.to_thread(
            self.client.delete_object, Bucket=self.bucket, Key=key
        )


def trace_store_from_env(env: dict[str, str] | None = None) -> TraceStoreProtocol:
    """按 TRACE_BACKEND 选择后端；缺省为本地目录。"""
    source = os.environ if env is None else env
    backend = (source.get("TRACE_BACKEND") or "local").lower()
    if backend in ("s3", "minio"):
        return S3TraceStore(
            endpoint=source.get("S3_ENDPOINT", "http://minio:9000"),
            bucket=source.get("S3_BUCKET", "traces"),
            access_key=source["S3_ACCESS_KEY"],
            secret_key=source["S3_SECRET_KEY"],
            region=source.get("S3_REGION", "us-east-1"),
            prefix=source.get("S3_PREFIX", ""),
        )
    return LocalTraceStore(source.get("TRACE_DIR", "./traces"))
