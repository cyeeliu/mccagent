"""API 端点契约测试：清单覆盖三组端点且无重复。"""

from eval_server.api import (
    INTERNAL_ENDPOINTS,
    PUBLIC_ENDPOINTS,
    SYSTEM_ENDPOINTS,
    all_endpoints,
)


def test_public_covers_core_resources() -> None:
    paths = {e.path for e in PUBLIC_ENDPOINTS}
    for required in (
        "/api/profiles",
        "/api/batches",
        "/api/runs",
        "/api/reports/compare",
        "/api/stats/dashboard",
        "/api/stats/trend",
    ):
        assert required in paths


def test_trace_and_stream_endpoints_exist() -> None:
    pairs = {(e.method, e.path) for e in PUBLIC_ENDPOINTS}
    assert ("GET", "/api/runs/{id}/trace") in pairs
    assert ("GET", "/api/runs/{id}/stream") in pairs


def test_internal_is_runner_only() -> None:
    assert INTERNAL_ENDPOINTS
    assert all(e.path.startswith("/internal/") for e in INTERNAL_ENDPOINTS)
    assert {e.method for e in INTERNAL_ENDPOINTS} == {"POST"}


def test_internal_lease_heartbeat_events_complete() -> None:
    pairs = {(e.method, e.path) for e in INTERNAL_ENDPOINTS}
    assert ("POST", "/internal/lease") in pairs
    assert ("POST", "/internal/runs/{id}/heartbeat") in pairs
    assert ("POST", "/internal/runs/{id}/events") in pairs
    assert ("POST", "/internal/runs/{id}/complete") in pairs


def test_system_probes_present() -> None:
    pairs = {(e.method, e.path) for e in SYSTEM_ENDPOINTS}
    assert ("GET", "/healthz/live") in pairs
    assert ("GET", "/healthz/ready") in pairs
    assert ("GET", "/metrics") in pairs


def test_no_duplicate_endpoints() -> None:
    seen = {(e.method, e.path) for e in all_endpoints()}
    assert len(seen) == len(all_endpoints())


def test_batches_support_submission_and_cancel() -> None:
    pairs = {(e.method, e.path) for e in PUBLIC_ENDPOINTS}
    assert ("POST", "/api/batches") in pairs
    assert ("POST", "/api/batches/{id}/cancel") in pairs
