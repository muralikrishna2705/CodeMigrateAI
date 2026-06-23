import { useState } from "react";

export default function AgentLog({ reports = [], compact = false }) {
  const [expanded, setExpanded] = useState(null);

  if (!reports.length) return null;

  // ── Compact mode: just colored dots in the tab bar ─────────────────────────
  if (compact) {
    return (
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {reports.map((r, i) => (
          <span
            key={i}
            title={`${r.agent}: ${r.summary}`}
            style={{
              width: 8, height: 8, borderRadius: "50%",
              background: r.status === "success" ? "var(--green)" : "var(--red)",
              boxShadow: r.status === "success"
                ? "0 0 5px var(--green)" : "0 0 5px var(--red)",
            }}
          />
        ))}
      </div>
    );
  }

  // ── Full mode: expandable cards ────────────────────────────────────────────
  return (
    <div style={st.log}>
      {reports.map((report, i) => (
        <div key={i} style={{
          ...st.card,
          borderLeft: `3px solid ${report.status === "success" ? "var(--green)" : "var(--red)"}`,
        }}>
          {/* Card header — clickable to expand */}
          <div style={st.cardHead} onClick={() => setExpanded(expanded === i ? null : i)}>
            <span style={{
              fontSize: 11, fontWeight: 700,
              color: report.status === "success" ? "var(--green)" : "var(--red)",
            }}>
              {report.status === "success" ? "✓" : "✗"}
            </span>
            <span style={st.agentName}>{report.agent}</span>
            <span style={st.chev}>{expanded === i ? "▲" : "▼"}</span>
          </div>

          {/* Summary */}
          <div style={st.summary}>{report.summary}</div>

          {/* Expandable detail */}
          {expanded === i && report.details && (
            <div style={st.detail}>
              {/* Scalar key-value pairs */}
              {Object.entries(report.details)
                .filter(([, v]) => !Array.isArray(v) && v != null && typeof v !== "object")
                .slice(0, 8)
                .map(([k, v]) => (
                  <div key={k} style={st.kv}>
                    <span style={st.k}>{k.replace(/_/g, " ")}</span>
                    <span style={st.v}>{String(v)}</span>
                  </div>
                ))}
              {/* Array fields */}
              {Object.entries(report.details)
                .filter(([, v]) => Array.isArray(v) && v.length > 0)
                .map(([k, v]) => (
                  <div key={k} style={st.kvList}>
                    <span style={st.k}>{k.replace(/_/g, " ")}:</span>
                    <ul style={st.ul}>
                      {v.slice(0, 5).map((item, j) => (
                        <li key={j} style={st.li}>{String(item)}</li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

const st = {
  log:      { display: "flex", flexDirection: "column", gap: 5 },
  card: {
    background: "var(--bg-raised)", borderRadius: "var(--radius)",
    border: "1px solid var(--border)", overflow: "hidden",
    fontSize: 11,
  },
  cardHead: {
    display: "flex", alignItems: "center", gap: 7,
    padding: "7px 10px", cursor: "pointer", userSelect: "none",
  },
  agentName: {
    flex: 1, fontFamily: "var(--font-code)",
    fontSize: 10, fontWeight: 600, color: "var(--text-normal)",
  },
  chev: { color: "var(--text-dim)", fontSize: 9 },
  summary: {
    padding: "0 10px 7px", color: "var(--text-muted)",
    fontSize: 10, lineHeight: 1.5,
  },
  detail: {
    padding: "8px 10px", background: "var(--bg-root)",
    borderTop: "1px solid var(--border)",
  },
  kv: { display: "flex", justifyContent: "space-between", gap: 6, padding: "2px 0" },
  k:  { color: "var(--text-dim)", textTransform: "capitalize", fontSize: 10 },
  v:  {
    color: "var(--text-normal)", fontFamily: "var(--font-code)",
    fontSize: 10, textAlign: "right",
  },
  kvList: { marginTop: 5, fontSize: 10 },
  ul:  { paddingLeft: 13, marginTop: 2 },
  li:  { color: "var(--text-muted)", marginBottom: 2, lineHeight: 1.4 },
};