import { useState } from "react";

export default function Output({ result, loading, language, version }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!result?.migrated_code) return;
    navigator.clipboard.writeText(result.migrated_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ── Loading state ──────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={st.panel}>
        <div style={st.header}>
          <span style={st.label}>Running Agent Pipeline…</span>
        </div>
        <div style={st.centered}>
          <div style={st.dots}>
            <span style={{ ...st.dot, animationDelay: "0s" }} />
            <span style={{ ...st.dot, animationDelay: ".2s" }} />
            <span style={{ ...st.dot, animationDelay: ".4s" }} />
          </div>
          <div style={st.steps}>
            <div style={st.step}>🔍 AnalyzerAgent — examining code structure…</div>
            <div style={st.step}>📋 PlannerAgent — building migration strategy…</div>
            <div style={st.step}>⚡ MigratorAgent — generating output with DeepSeek-Coder-V2…</div>
          </div>
          <style>{`
            @keyframes bounce {
              0%,80%,100% { transform:scale(.5); opacity:.3 }
              40%          { transform:scale(1);  opacity:1   }
            }
          `}</style>
        </div>
      </div>
    );
  }

  // ── Empty state ────────────────────────────────────────────────────────────
  if (!result) {
    return (
      <div style={st.panel}>
        <div style={st.header}>
          <span style={st.label}>{language.toUpperCase()} {version} — Output</span>
        </div>
        <div style={st.centered}>
          <span style={st.emptyIcon}>⟳</span>
          <p style={st.emptyText}>Configure your migration and click "Run Migration"</p>
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
      </div>
    );
  }

  // ── Result ─────────────────────────────────────────────────────────────────
  const lines = result.migrated_code.split("\n").length;
  return (
    <div style={st.panel}>
      <div style={st.header}>
        <span style={st.label}>{language.toUpperCase()} {version} — Output</span>
        <div style={st.actions}>
          <span style={{
            ...st.tag,
            color:       result.success ? "var(--green)"  : "var(--amber)",
            borderColor: result.success ? "rgba(74,222,128,.3)" : "rgba(251,191,36,.3)",
            background:  result.success ? "rgba(74,222,128,.08)" : "rgba(251,191,36,.08)",
          }}>
            {result.success ? "✓ Success" : "⚠ Partial"}
          </span>
          <span style={st.lineCount}>{lines} lines</span>
          <button
            style={st.btn}
            onClick={handleCopy}
            onMouseEnter={e => e.target.style.color = "var(--accent)"}
            onMouseLeave={e => e.target.style.color = "var(--text-muted)"}
          >
            {copied ? "✓ Copied!" : "Copy"}
          </button>
        </div>
      </div>
      <pre style={st.pre}>{result.migrated_code}</pre>
    </div>
  );
}

const st = {
  panel:  { display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" },
  header: {
    display: "flex", alignItems: "center", gap: 10,
    padding: "8px 16px", background: "var(--bg-raised)",
    borderBottom: "1px solid var(--border)", minHeight: 38, flexShrink: 0,
  },
  label: {
    flex: 1, fontSize: 10, fontWeight: 700, textTransform: "uppercase",
    letterSpacing: "1px", color: "var(--text-muted)", fontFamily: "var(--font-code)",
  },
  actions: { display: "flex", alignItems: "center", gap: 10 },
  tag: {
    padding: "2px 8px", borderRadius: 4, fontSize: 10,
    fontWeight: 700, border: "1px solid",
  },
  lineCount: { fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--font-code)" },
  btn: {
    padding: "2px 9px", background: "var(--bg-overlay)",
    border: "1px solid var(--border-mid)", borderRadius: 4,
    color: "var(--text-muted)", fontSize: 10,
    fontFamily: "var(--font-ui)", cursor: "pointer", transition: "color .15s",
  },
  pre: {
    flex: 1, background: "var(--bg-root)", color: "var(--text-bright)",
    fontFamily: "var(--font-code)", fontSize: 12.5, lineHeight: 1.75,
    padding: "14px 18px", overflow: "auto", whiteSpace: "pre",
    margin: 0,
  },
  centered: {
    flex: 1, display: "flex", flexDirection: "column",
    alignItems: "center", justifyContent: "center",
    background: "var(--bg-root)", gap: 16,
  },
  dots: { display: "flex", gap: 8 },
  dot: {
    display: "inline-block", width: 10, height: 10,
    background: "var(--accent)", borderRadius: "50%",
    animation: "bounce 1.3s ease-in-out infinite",
  },
  steps: { display: "flex", flexDirection: "column", gap: 8 },
  step:  { fontSize: 12, color: "var(--text-muted)" },
  emptyIcon: {
    fontSize: 44, color: "var(--border-mid)",
    display: "inline-block", animation: "spin 12s linear infinite",
  },
  emptyText: { fontSize: 13, color: "var(--text-dim)" },
};