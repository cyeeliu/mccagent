// 与 schemas/trace-event.schema.json 对齐的运行时契约（schema_version=1）。
// 所有网络数据先过 Zod 校验：类型错误在网络层暴露，而不是渗透到 UI 变成渲染异常。
import { z } from "zod";

const base = { schema_version: z.literal(1), seq: z.number(), ts: z.number() };

export const TraceEventSchema = z.discriminatedUnion("type", [
  z.object({ ...base, type: z.literal("run_started"), run_id: z.string(),
             case_id: z.string(), task: z.string(), model: z.string().nullish() }),
  z.object({ ...base, type: z.literal("llm_request"), turn: z.number(),
             input_tokens: z.number().default(0) }),
  z.object({ ...base, type: z.literal("llm_chunk"), turn: z.number(), delta: z.string() }),
  z.object({ ...base, type: z.literal("tool_call"), turn: z.number(),
             call_id: z.string(), tool_name: z.string(), args: z.unknown() }),
  z.object({ ...base, type: z.literal("tool_result"), call_id: z.string(),
             output: z.string(), is_error: z.boolean().default(false),
             duration_ms: z.number().default(0) }),
  z.object({ ...base, type: z.literal("run_finished"),
             status: z.enum(["passed", "failed", "error", "timeout"]),
             score: z.number().nullish(), cost_usd: z.number().nullish(),
             turns: z.number().default(0) }),
]);

export type TraceEvent = z.infer<typeof TraceEventSchema>;

/** 未知事件类型降级为占位而不是崩溃：前端必须假设后端可能部署了更新的 schema。 */
export type MaybeEvent = TraceEvent | { type: "unknown"; seq: number; raw: unknown };

export function parseEvent(raw: unknown): MaybeEvent {
  const r = TraceEventSchema.safeParse(raw);
  if (r.success) return r.data;
  const seq = typeof (raw as { seq?: number })?.seq === "number" ? (raw as { seq: number }).seq : -1;
  return { type: "unknown", seq, raw };
}

export const RunSchema = z.object({
  id: z.string(),
  batch_id: z.string(),
  case_id: z.string(),
  status: z.string(),
  score: z.number().nullish(),
  cost_usd: z.number().nullish(),
  turns: z.number().nullish(),
});
export type Run = z.infer<typeof RunSchema>;

export const BatchSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.string(),
  passed: z.number(),
  total: z.number(),
  cost_usd: z.number().nullish(),
  pass_rate: z.number().nullish(),
  created_at: z.string(),
});
export type Batch = z.infer<typeof BatchSchema>;

export const BatchDetailSchema = z.object({
  id: z.string(),
  name: z.string(),
  status: z.string(),
  parallelism: z.number(),
  created_at: z.string(),
  run_stats: z.record(z.number()),
});
export type BatchDetail = z.infer<typeof BatchDetailSchema>;

export const DashboardSchema = z.object({
  total_runs: z.number(),
  total_batches: z.number(),
  total_cost_usd: z.number(),
  pass_rate: z.number(),
  runs_last_24h: z.number(),
  queue_depth: z.number(),
  active_runs: z.number(),
  status_breakdown: z.record(z.number()),
});
export type Dashboard = z.infer<typeof DashboardSchema>;

export const TrendSchema = z.object({
  time: z.string(),
  passed: z.number(),
  total: z.number(),
  cost_usd: z.number(),
  pass_rate: z.number(),
});
export type TrendPoint = z.infer<typeof TrendSchema>;

const CaseVerdictSchema = z.object({
  case_id: z.string(),
  status_a: z.string(),
  status_b: z.string(),
  score_a: z.number().nullish(),
  score_b: z.number().nullish(),
  run_a: z.string(),
  run_b: z.string(),
  verdict: z.enum(["regression", "improvement", "same"]),
});

export const CompareReportSchema = z.object({
  comparability_warnings: z.array(z.string()),
  summary: z.object({
    total: z.number(),
    passed_a: z.number(),
    passed_b: z.number(),
    pass_rate_a: z.number(),
    pass_rate_b: z.number(),
    regressions: z.number(),
    improvements: z.number(),
  }),
  cases: z.array(CaseVerdictSchema),
});
export type CompareReport = z.infer<typeof CompareReportSchema>;
export type CaseVerdict = z.infer<typeof CaseVerdictSchema>;
