import { describe, expect, it } from "vitest";
import {
  CompareReportSchema,
  DashboardSchema,
  TraceEventSchema,
  parseEvent,
} from "./schemas";

const base = { schema_version: 1, seq: 5, ts: 1_700_000_000 };

describe("TraceEventSchema", () => {
  it("接受六种已知事件类型", () => {
    const samples = [
      { ...base, type: "run_started", run_id: "r", case_id: "c", task: "t", model: null },
      { ...base, type: "llm_request", turn: 0 },
      { ...base, type: "llm_chunk", turn: 0, delta: "hi" },
      { ...base, type: "tool_call", turn: 0, call_id: "x", tool_name: "bash", args: {} },
      { ...base, type: "tool_result", call_id: "x", output: "ok" },
      { ...base, type: "run_finished", status: "passed" },
    ];
    for (const s of samples) {
      expect(TraceEventSchema.safeParse(s).success).toBe(true);
    }
  });

  it("带默认值的字段缺失时校验仍通过并补全", () => {
    const r = TraceEventSchema.parse({ ...base, type: "tool_result", call_id: "x", output: "o" });
    expect(r).toMatchObject({ is_error: false, duration_ms: 0 });
  });

  it("run_finished 的 status 只允许四个终态", () => {
    const bad = { ...base, type: "run_finished", status: "running" };
    expect(TraceEventSchema.safeParse(bad).success).toBe(false);
  });
});

describe("parseEvent 未知事件降级", () => {
  it("未知类型降级为占位并保留 seq", () => {
    const e = parseEvent({ ...base, type: "future_event" });
    expect(e.type).toBe("unknown");
    expect(e).toMatchObject({ seq: 5 });
  });

  it("缺 seq 的坏数据降级为 -1", () => {
    const e = parseEvent({ type: "future_event" });
    expect(e).toMatchObject({ type: "unknown", seq: -1 });
  });

  it("字段缺失的已知类型同样降级而不是抛错", () => {
    const e = parseEvent({ ...base, type: "llm_chunk", turn: 0 });
    expect(e.type).toBe("unknown");
  });
});

describe("聚合响应 schema", () => {
  it("DashboardSchema 匹配服务端形状", () => {
    const r = DashboardSchema.safeParse({
      total_runs: 10, total_batches: 2, total_cost_usd: 1.5, pass_rate: 0.8,
      runs_last_24h: 4, queue_depth: 1, active_runs: 2,
      status_breakdown: { passed: 8, failed: 2 },
    });
    expect(r.success).toBe(true);
  });

  it("CompareReportSchema 匹配服务端形状", () => {
    const r = CompareReportSchema.safeParse({
      comparability_warnings: ["model 不同（a vs b）"],
      summary: {
        total: 2, passed_a: 1, passed_b: 2, pass_rate_a: 0.5, pass_rate_b: 1,
        regressions: 0, improvements: 1,
      },
      cases: [
        {
          case_id: "c1", status_a: "failed", status_b: "passed",
          score_a: 0, score_b: 1, run_a: "u1", run_b: "u2", verdict: "improvement",
        },
      ],
    });
    expect(r.success).toBe(true);
  });
});
