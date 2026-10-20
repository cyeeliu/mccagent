// 实时与历史统一视图：历史分页拉全 + SSE 续接 + 按 seq 合并去重。
// run 的三种状态（已结束 / 正在跑 / 看着看着跑完）都被封装在这个 Hook 里，组件无感知。
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchRun, fetchTracePage, subscribeRun } from "../../api/client";
import type { MaybeEvent } from "../../api/schemas";

const LIVE_STATUSES = ["queued", "leased", "running"];

/** seq 由后端单调分配：用它做精确对齐，历史与实时重叠部分去重后按序输出。 */
export function mergeBySeq(history: MaybeEvent[], live: MaybeEvent[]): MaybeEvent[] {
  const seen = new Set<number>();
  const merged: MaybeEvent[] = [];
  for (const e of [...history, ...live]) {
    if (e.seq >= 0 && seen.has(e.seq)) continue;
    seen.add(e.seq);
    merged.push(e);
  }
  return merged.sort((a, b) => a.seq - b.seq);
}

export function useTrace(runId: string) {
  const { data: run, refetch } = useQuery({
    queryKey: ["runs", runId],
    queryFn: () => fetchRun(runId),
    refetchInterval: (q) =>
      LIVE_STATUSES.includes(q.state.data?.status ?? "") ? 2000 : false,
  });
  const isLive = LIVE_STATUSES.includes(run?.status ?? "");

  const [history, setHistory] = useState<MaybeEvent[]>([]);
  const [live, setLive] = useState<MaybeEvent[]>([]);
  const bufRef = useRef<MaybeEvent[]>([]);

  // 历史段：分页拉全；run 结束时再拉一次，保证终态事件完整落盘
  useEffect(() => {
    let cancelled = false;
    setHistory([]);
    (async () => {
      const all: MaybeEvent[] = [];
      let offset: number | null = 0;
      while (offset !== null) {
        const page = await fetchTracePage(runId, offset);
        all.push(...page.events);
        offset = page.nextOffset;
      }
      if (!cancelled) setHistory(all);
    })().catch(console.error);
    return () => { cancelled = true; };
  }, [runId, isLive]);

  // 实时段：SSE + 100ms 节流合帧，避免高频 chunk 逐条触发渲染
  useEffect(() => {
    if (!isLive) return;
    const flush = setInterval(() => {
      if (bufRef.current.length) {
        const b = bufRef.current;
        bufRef.current = [];
        setLive((prev) => [...prev, ...b]);
      }
    }, 100);
    const unsubscribe = subscribeRun(
      runId,
      (e) => {
        bufRef.current.push(e);
        if (e.type === "run_finished") void refetch();
      },
      () => void refetch(), // lagged：触发 run 重查 → isLive 翻转 → 历史段全量重拉补偿
    );
    return () => { unsubscribe(); clearInterval(flush); };
  }, [runId, isLive, refetch]);

  const events = useMemo(() => mergeBySeq(history, live), [history, live]);

  return { run, events, isLive };
}
