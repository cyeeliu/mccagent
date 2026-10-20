// 批次列表 + 批次详情：运行中自动轮询，跑完自动停。
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { fetchBatch, fetchBatches, fetchRuns } from "../../api/client";

const LIVE_RUN_STATUSES = ["queued", "leased", "running"];

export function BatchesPage() {
  const { data: batches, isLoading, error } = useQuery({
    queryKey: ["batches"],
    queryFn: fetchBatches,
    refetchInterval: (q) =>
      q.state.data?.some((b) => b.status === "running") ? 2000 : 15000,
  });

  if (isLoading) return <p className="empty">加载中…</p>;
  if (error) return <p className="empty">出错了：{String(error)}</p>;

  return (
    <main className="page">
      <h1>评测批次</h1>
      {batches?.length === 0 && (
        <p className="empty">还没有批次，先通过 API 创建一个吧</p>
      )}
      <table className="table">
        <thead>
          <tr><th>名称</th><th>状态</th><th>通过</th><th>成本</th><th>创建时间</th></tr>
        </thead>
        <tbody>
          {batches?.map((b) => (
            <tr key={b.id}>
              <td><Link to={`/batches/${b.id}`}>{b.name}</Link></td>
              <td><span className={`badge ${b.status}`}>{b.status}</span></td>
              <td>
                {b.total > 0
                  ? `${b.passed} / ${b.total} (${((b.passed / b.total) * 100).toFixed(0)}%)`
                  : "0 / 0"}
              </td>
              <td>{b.cost_usd != null ? `$${b.cost_usd.toFixed(4)}` : "—"}</td>
              <td>{new Date(b.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

export function BatchDetailPage() {
  const { batchId } = useParams();
  const { data: batch } = useQuery({
    queryKey: ["batches", batchId],
    queryFn: () => fetchBatch(batchId!),
  });
  const { data: runs } = useQuery({
    queryKey: ["batches", batchId, "runs"],
    queryFn: () => fetchRuns(batchId!),
    refetchInterval: (q) =>
      q.state.data?.some((r) => LIVE_RUN_STATUSES.includes(r.status))
        ? 1500
        : false,
  });

  return (
    <main className="page">
      <h1>
        {batch?.name ?? `批次 ${batchId?.slice(0, 8)}`}
        {batch && <span className={`badge ${batch.status}`}>{batch.status}</span>}
      </h1>
      {batch && (
        <p className="muted">
          并发上限 {batch.parallelism} · 创建于 {new Date(batch.created_at).toLocaleString()} ·{" "}
          {Object.entries(batch.run_stats).map(([s, n]) => `${s} ${n}`).join(" · ")}
        </p>
      )}
      <table className="table">
        <thead>
          <tr><th>Case</th><th>状态</th><th>分数</th><th>成本</th><th>轮数</th><th></th></tr>
        </thead>
        <tbody>
          {runs?.map((r) => (
            <tr key={r.id} className={r.status === "failed" || r.status === "error" ? "row-failed" : ""}>
              <td>{r.case_id}</td>
              <td><span className={`badge ${r.status}`}>{r.status}</span></td>
              <td>{r.score != null ? r.score.toFixed(2) : "—"}</td>
              <td>{r.cost_usd != null ? `$${r.cost_usd.toFixed(4)}` : "—"}</td>
              <td>{r.turns ?? "—"}</td>
              <td><Link to={`/runs/${r.id}`}>轨迹 →</Link></td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}
