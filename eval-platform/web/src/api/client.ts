// API 客户端：每个函数都用 Zod schema 验证响应，类型不符立刻在边界抛错。
import { z } from "zod";
import {
  BatchSchema,
  BatchDetailSchema,
  RunSchema,
  DashboardSchema,
  TrendSchema,
  CompareReportSchema,
  parseEvent,
  type MaybeEvent,
} from "./schemas";

async function get<T>(url: string, schema: z.ZodType<T>): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return schema.parse(await res.json());
}

export const fetchBatches = () =>
  get(`/api/batches`, z.object({ items: z.array(BatchSchema) })).then((r) => r.items);

export const fetchBatch = (batchId: string) =>
  get(`/api/batches/${batchId}`, BatchDetailSchema);

export const fetchRuns = (batchId: string) =>
  get(`/api/runs?batch=${batchId}`, z.object({ items: z.array(RunSchema) })).then((r) => r.items);

export const fetchRun = (runId: string) => get(`/api/runs/${runId}`, RunSchema);

export async function fetchTracePage(runId: string, offset: number) {
  const res = await fetch(`/api/runs/${runId}/trace?offset=${offset}&limit=500`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const page = z
    .object({ events: z.array(z.unknown()), total: z.number(), next_offset: z.number().nullable() })
    .parse(await res.json());
  return { events: page.events.map(parseEvent), nextOffset: page.next_offset };
}

export const fetchDashboard = () =>
  get(`/api/stats/dashboard`, DashboardSchema);

export const fetchTrend = (days = 30, groupBy = "day") =>
  get(
    `/api/stats/trend?days=${days}&group_by=${groupBy}`,
    z.object({ points: z.array(TrendSchema) }),
  ).then((r) => r.points);

export const fetchCompare = (a: string, b: string) =>
  get(`/api/reports/compare?a=${a}&b=${b}`, CompareReportSchema);

/** SSE 订阅：onEvent 逐条回调；掉帧时后端发 lagged 信号，由调用方走 REST 全量补偿。 */
export function subscribeRun(
  runId: string,
  onEvent: (e: MaybeEvent) => void,
  onLagged: () => void,
): () => void {
  const es = new EventSource(`/api/runs/${runId}/stream`);
  es.onmessage = (msg) => onEvent(parseEvent(JSON.parse(msg.data)));
  es.addEventListener("lagged", onLagged);
  es.onerror = () => {
    // EventSource 会自动重连，但服务端不支持按 Last-Event-ID 续发——
    // 断线窗口内的事件永久丢失，须走 REST 全量补齐（与 lagged 同路径）
    onLagged();
  };
  return () => es.close();
}
