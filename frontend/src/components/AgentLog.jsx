import { useState } from "react";

export default function AgentLog({ reports = [], compact = false }) {
  const [expanded, setExpanded] = useState(null);

  if (!reports.length) return null;

  // ── Compact: status dots in the view bar ───────────────────────────────────
  if (compact) {
    return (
      <div className="agent-dots">
        {reports.map((r, i) => {
          const ok = r.status === "success";
          return (
            <span
              key={i}
              className="dot"
              title={`${r.agent}: ${r.summary}`}
              style={{
                background: ok ? "var(--success)" : "var(--danger)",
                boxShadow: `0 0 6px ${ok ? "var(--success)" : "var(--danger)"}`,
              }}
            />
          );
        })}
      </div>
    );
  }

  // ── Full: expandable cards ─────────────────────────────────────────────────
  return (
    <div className="pipeline-list">
      {reports.map((report, i) => {
        const ok = report.status === "success";
        const open = expanded === i;
        const hasDetail = Boolean(report.details && Object.keys(report.details).length);

        return (
          <div key={i} className={"agent-card" + (ok ? "" : " agent-card--fail")}>
            <button
              className="agent-head"
              onClick={() => setExpanded(open ? null : i)}
              aria-expanded={open}
              disabled={!hasDetail}
            >
              <span className="agent-glyph" aria-hidden="true">{ok ? "✓" : "✗"}</span>
              <span className="agent-name">{report.agent}</span>
              {hasDetail && (
                <span className={"agent-chev" + (open ? " agent-chev--open" : "")} aria-hidden="true">▾</span>
              )}
            </button>

            <p className="agent-summary">{report.summary}</p>

            {open && hasDetail && (
              <div className="agent-detail">
                <dl>
                  {Object.entries(report.details)
                    .filter(([, v]) => v != null && typeof v !== "object")
                    .slice(0, 8)
                    .map(([k, v]) => (
                      <div className="kv" key={k}>
                        <dt>{k.replace(/_/g, " ")}</dt>
                        <dd>{String(v)}</dd>
                      </div>
                    ))}
                </dl>

                {Object.entries(report.details)
                  .filter(([, v]) => Array.isArray(v) && v.length > 0)
                  .map(([k, v]) => (
                    <div className="kv-list" key={k}>
                      <span>{k.replace(/_/g, " ")}</span>
                      <ul>
                        {v.slice(0, 5).map((item, j) => (
                          <li key={j}>{String(item)}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
