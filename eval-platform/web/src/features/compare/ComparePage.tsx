/**
 * 批次对比页面：URL 参数驱动，链接可分享。
 *
 * URL: /compare?a=<batch_uuid>&b=<batch_uuid>
 *
 * 功能：
 *   - 总结卡：A / B 通过率、回归数、改进数
 *   - 可比性警告（model / harness / sandbox 镜像不同时由后端给出）
 *   - Case 列表：按 verdict 过滤（regression / improvement / same）
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { fetchCompare } from "../../api/client";
import type { CaseVerdict } from "../../api/schemas";

function SummaryCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string | number;
  accent?: string;
}) {
  return (
    <div className="stat-card" style={{ borderTop: `3px solid ${accent ?? "#6366f1"}` }}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  );
}

const VERDICT_COLOR: Record<string, string> = {
  regression: "#ef4444",
  improvement: "#10b981",
  same: "#9ca3af",
};

function VerdictBadge({ v }: { v: string }) {
  return (
    <span className="badge" style={{ background: VERDICT_COLOR[v] ?? "#9ca3af" }}>
      {v === "regression" ? "↓ 回归" : v === "improvement" ? "↑ 改进" : "= 不变"}
    </span>
  );
}

function CaseRow({ c }: { c: CaseVerdict }) {
  return (
    <tr>
      <td className="mono">{c.case_id}</td>
      <td>
        <VerdictBadge v={c.verdict} />
      </td>
      <td>{c.status_a}</td>
      <td>{c.status_b}</td>
      <td>{c.score_a != null ? c.score_a.toFixed(3) : "—"}</td>
      <td>{c.score_b != null ? c.score_b.toFixed(3) : "—"}</td>
      <td>
        <Link to={`/runs/${c.run_a}`} target="_blank" rel="noreferrer">
          A
        </Link>
        {" · "}
        <Link to={`/runs/${c.run_b}`} target="_blank" rel="noreferrer">
          B
        </Link>
      </td>
    </tr>
  );
}

export function ComparePage() {
  const [params] = useSearchParams();
  const a = params.get("a") ?? "";
  const b = params.get("b") ?? "";
  const [filter, setFilter] = useState<"all" | "regression" | "improvement" | "same">("all");

  const { data, isLoading, isError } = useQuery({
    queryKey: ["compare", a, b],
    queryFn: () => fetchCompare(a, b),
    enabled: Boolean(a && b),
  });

  if (!a || !b) {
    return (
      <div className="page">
        <div className="empty-state">
          <p>请在 URL 中提供两个批次 ID：</p>
          <code>/compare?a=&lt;batch_id&gt;&amp;b=&lt;batch_id&gt;</code>
        </div>
      </div>
    );
  }

  if (isLoading) return <div className="loading">计算对比中…</div>;
  if (isError || !data)
    return <div className="error">加载失败，请检查批次 ID 是否存在。</div>;

  const { summary, cases, comparability_warnings } = data;
  const filtered = filter === "all" ? cases : cases.filter((c) => c.verdict === filter);

  const passRateA = (summary.pass_rate_a * 100).toFixed(1);
  const passRateB = (summary.pass_rate_b * 100).toFixed(1);
  const delta = summary.pass_rate_b - summary.pass_rate_a;
  const deltaStr = `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)}%`;

  return (
    <div className="page">
      <h2 className="page-title">批次对比</h2>
      <p className="muted mono">
        A: {a.slice(0, 8)}… &nbsp;vs&nbsp; B: {b.slice(0, 8)}…
      </p>

      {comparability_warnings.length > 0 && (
        <div className="warning-box">
          <strong>⚠ 可比性提示</strong>
          <ul>
            {comparability_warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="stat-grid">
        <SummaryCard label="A 通过率" value={`${passRateA}%`} accent="#6366f1" />
        <SummaryCard label="B 通过率" value={`${passRateB}%`} accent="#6366f1" />
        <SummaryCard
          label="通过率变化"
          value={deltaStr}
          accent={delta >= 0 ? "#10b981" : "#ef4444"}
        />
        <SummaryCard label="回归 / 改进" value={`${summary.regressions} / ${summary.improvements}`} />
      </div>

      <div className="filter-row">
        {(["all", "regression", "improvement", "same"] as const).map((v) => (
          <button
            key={v}
            className={`filter-btn${filter === v ? " active" : ""}`}
            onClick={() => setFilter(v)}
          >
            {v === "all"
              ? `全部 (${cases.length})`
              : v === "regression"
              ? `回归 (${summary.regressions})`
              : v === "improvement"
              ? `改进 (${summary.improvements})`
              : `不变 (${cases.length - summary.regressions - summary.improvements})`}
          </button>
        ))}
      </div>

      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Verdict</th>
              <th>状态 A</th>
              <th>状态 B</th>
              <th>Score A</th>
              <th>Score B</th>
              <th>Trace</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <CaseRow key={c.case_id} c={c} />
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="empty">
                  该过滤条件下没有数据
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
