"""指标测试：记录入口、导出格式与实例隔离。"""

from eval_server.metrics import Metrics


def test_render_contains_registered_metrics() -> None:
    m = Metrics()
    text = m.render()
    assert "http_requests_total" in text
    assert "eval_runs_total" in text
    assert "eval_queue_depth" in text
    assert "eval_active_runs" in text
    assert "trace_store_append_total" in text
    assert "events_ingested_total" in text


def test_http_request_records_labels() -> None:
    m = Metrics()
    m.http_request("GET", "/api/runs", 200, 0.05)
    m.http_request("GET", "/api/runs", 500, 0.2)
    text = m.render()
    assert 'method="GET"' in text
    assert 'path="/api/runs"' in text
    assert 'status="200"' in text
    assert 'status="500"' in text


def test_run_completed_with_and_without_cost() -> None:
    m = Metrics()
    m.run_completed("passed", 12.5, 0.03)
    m.run_completed("error", 1.0, None)
    text = m.render()
    assert 'status="passed"' in text
    assert 'status="error"' in text
    assert "eval_run_cost_usd" in text


def test_update_queue_gauges_sets_values() -> None:
    m = Metrics()
    m.update_queue_gauges(queued=7, active=3)
    text = m.render()
    assert "eval_queue_depth 7.0" in text
    assert "eval_active_runs 3.0" in text


def test_trace_and_events_counters() -> None:
    m = Metrics()
    m.trace_append(lines=42)
    m.events_ingested_inc(accepted=40, rejected=2)
    text = m.render()
    assert "trace_store_append_total 1.0" in text
    assert "trace_store_lines_per_batch" in text
    assert 'outcome="accepted"' in text
    assert 'outcome="rejected"' in text


def test_instances_have_independent_registries() -> None:
    m1 = Metrics()
    m2 = Metrics()
    m1.http_request("GET", "/x", 200, 0.01)
    assert "path=\"/x\"" in m1.render()
    assert "path=\"/x\"" not in m2.render()
