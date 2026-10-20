// 轨迹查看器：时间线 + 详情双栏；只看失败过滤；未知事件降级占位。
import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import type { MaybeEvent } from "../../api/schemas";
import { useTrace } from "./useTrace";

const ICONS: Record<string, string> = {
  run_started: "▶", llm_request: "🧠", llm_chunk: "💬",
  tool_call: "🔧", tool_result: "↩", run_finished: "■", unknown: "?",
};

function describe(e: MaybeEvent): string {
  switch (e.type) {
    case "run_started": return `开始 · ${e.task.slice(0, 80)}`;
    case "llm_request": return `第 ${e.turn + 1} 轮请求 · ${e.input_tokens} tokens`;
    case "llm_chunk": return e.delta.slice(0, 100);
    case "tool_call": return `${e.tool_name}(${JSON.stringify(e.args ?? {}).slice(0, 80)})`;
    case "tool_result": return (e.is_error ? "✗ " : "✓ ") + e.output.slice(0, 90) + ` · ${e.duration_ms}ms`;
    case "run_finished": return `结束 · ${e.status} · $${(e.cost_usd ?? 0).toFixed(4)} · ${e.turns} 轮`;
    default: return "未知事件类型（schema 版本可能更新）";
  }
}

export function TracePage() {
  const { runId } = useParams();
  const { run, events, isLive } = useTrace(runId!);
  const [selectedSeq, setSelectedSeq] = useState<number | null>(null);
  const [onlyErrors, setOnlyErrors] = useState(false);

  const visible = useMemo(
    () => events.filter((e) => !onlyErrors || (e.type === "tool_result" && e.is_error)),
    [events, onlyErrors],
  );
  const selected = events.find((e) => e.seq === selectedSeq) ?? null;

  return (
    <div className="trace-layout">
      <main className="timeline-pane">
        <header className="run-header">
          <h1>
            {run?.case_id ?? runId}
            {isLive && <span className="badge live">LIVE</span>}
            {run && !isLive && <span className={`badge ${run.status}`}>{run.status}</span>}
            {run && (run.retries ?? 0) > 0 && isLive && (
              <span className="badge retry" title={`租约过期被重排，当前第 ${(run.retries ?? 0) + 1} 次尝试`}>
                ↻ 重跑中
              </span>
            )}
          </h1>
          <label>
            <input type="checkbox" checked={onlyErrors}
                   onChange={(e) => setOnlyErrors(e.target.checked)} />
            只看失败
          </label>
          <span className="meta">{events.length} 个事件</span>
        </header>
        <div className="timeline">
          {visible.map((e, i) => (
            <div
              key={e.seq >= 0 ? `s${e.seq}` : `u${i}`}
              className={[
                "event-row",
                e.type,
                e.type === "tool_result" && e.is_error ? "failed" : "",
                e.seq === selectedSeq ? "selected" : "",
              ].join(" ")}
              onClick={() => setSelectedSeq(e.seq)}
            >
              <span className="icon">{ICONS[e.type]}</span>
              <span className="summary">{describe(e)}</span>
            </div>
          ))}
          {visible.length === 0 && <p className="empty">暂无事件</p>}
        </div>
      </main>
      <aside className="inspector">
        <h2>事件详情</h2>
        {selected ? (
          <pre>{JSON.stringify(selected.type === "unknown" ? selected.raw : selected, null, 2)}</pre>
        ) : (
          <p className="empty">点击左侧事件查看原始数据</p>
        )}
      </aside>
    </div>
  );
}
