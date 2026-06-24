import { useState } from "react";

export default function Output({ result, streamBuffer, loading, language, version, agentProgress }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!result?.migrated_code) return;
    navigator.clipboard.writeText(result.migrated_code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ── Loading state ──────────────────────────────────────────────────────────
  if (loading) {
    // If streaming, show progress with agent status and code buffer
    if (agentProgress && agentProgress.length > 0) {
      return (
        <div style={st.panel}>
          <div style={st.header}>
            <span style={st.label}>Running Agent Pipeline…</span>
          </div>
          <div style={st.centered}>
            <div style={st.agentProgress}>
              {agentProgress.map((a, i) => (
                <div key={i} style={st.agentProgressItem}>
                  <span style={{
                    ...st.agentStatusDot,
                    background: a.status === "complete" ? "var(--green)" : "var(--accent)",
                  }} />
                  <span style={{ fontSize: 12, color: "var(--text-normal)" }}>
                    {a.agent} {a.status === "complete" ? "✓" : "⟳"}
                    {a.message && <span style={{ color: "var(--text-muted)", marginLeft: 8 }}>{a.message}</span>}
                  </span>
                </div>
              ))}
            </div>
            {streamBuffer && (
              <div style={st.streamingCode}>
                <pre style={st.pre}>{streamBuffer}</pre>
              </div>
            )}
          </div>
        </div>
      );
    }

    // Non-streaming loading state
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
            <div style={st.step}>⚡ MigratorAgent — composing prompt and generating output…</div>
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
  if (!result && !streamBuffer) {
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

  // ── Result (or streaming buffer) ─────────────────────────────────────────────
  const displayCode = result?.migrated_code || streamBuffer || "";
  const planSummary = result?.inline_plan || "";
  const validationState = getValidationState(result?.validation_result);
  const lines = displayCode.split("\n").length;

  return (
    <div style={st.panel}>
      <div style={st.header}>
        <span style={st.label}>{language.toUpperCase()} {version} — Output</span>
        <div style={st.actions}>
          {result && (
            <span style={{
              ...st.tag,
              color:       result.success ? "var(--green)"  : "var(--amber)",
              borderColor: result.success ? "rgba(74,222,128,.3)" : "rgba(251,191,36,.3)",
              background:  result.success ? "rgba(74,222,128,.08)" : "rgba(251,191,36,.08)",
            }}>
              {result.success ? "✓ Success" : "⚠ Partial"}
            </span>
          )}
          {validationState && (
            <span style={{
              ...st.tag,
              color: validationState.color,
              borderColor: validationState.border,
              background: validationState.background,
            }}>
              {validationState.label}
            </span>
          )}
          <span style={st.lineCount}>{lines} lines</span>
          <button
            style={st.btn}
            onClick={handleCopy}
            onMouseEnter={e => e.target.style.color = "var(--accent)"}
            onMouseLeave={e => e.target.style.color = "var(--text-muted)"}
            disabled={!displayCode}
          >
            {copied ? "✓ Copied!" : "Copy"}
          </button>
        </div>
      </div>
      {planSummary && (
        <div style={st.plan}>
          <span style={st.planLabel}>Plan</span>
          <span style={st.planText}>{planSummary}</span>
        </div>
      )}
      {validationState?.items.length > 0 && (
        <details style={st.validation}>
          <summary style={st.validationSummary}>Validation details</summary>
          <div style={st.validationList}>
            {validationState.items.map((item, index) => (
              <div key={index} style={st.validationItem}>
                {formatIssue(item)}
              </div>
            ))}
          </div>
        </details>
      )}
      <pre style={st.pre}>{displayCode}</pre>
    </div>
  );
}

function getValidationState(validation) {
  if (!validation) return null;

  const valid = Boolean(validation.valid ?? validation.syntax_valid);
  const syntaxErrors = validation.syntax_errors || {};
  const errors = validation.errors || syntaxErrors.errors || [];
  const warnings = validation.warnings || syntaxErrors.warnings || [];
  const items = [...errors, ...warnings];

  if (!valid) {
    return {
      label: "Validation failed",
      color: "var(--red)",
      border: "rgba(248,113,113,.3)",
      background: "rgba(248,113,113,.08)",
      items,
    };
  }

  if (warnings.length > 0) {
    return {
      label: "Validation warnings",
      color: "var(--amber)",
      border: "rgba(251,191,36,.3)",
      background: "rgba(251,191,36,.08)",
      items,
    };
  }

  return {
    label: "Validation OK",
    color: "var(--green)",
    border: "rgba(74,222,128,.3)",
    background: "rgba(74,222,128,.08)",
    items: [],
  };
}

function formatIssue(issue) {
  const location = issue.line ? `${issue.line}:${issue.column || 0} ` : "";
  return `${location}${issue.message || String(issue)}`;
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
  plan: {
    display: "flex", gap: 10, alignItems: "flex-start",
    padding: "9px 16px", background: "var(--bg-base)",
    borderBottom: "1px solid var(--border)", flexShrink: 0,
  },
  planLabel: {
    fontSize: 10, fontWeight: 700, textTransform: "uppercase",
    color: "var(--accent)", fontFamily: "var(--font-code)",
  },
  planText: { fontSize: 12, color: "var(--text-normal)", lineHeight: 1.45 },
  validation: {
    padding: "8px 16px", background: "var(--bg-base)",
    borderBottom: "1px solid var(--border)", flexShrink: 0,
  },
  validationSummary: {
    color: "var(--text-muted)", cursor: "pointer", fontSize: 11,
    fontFamily: "var(--font-ui)",
  },
  validationList: {
    display: "flex", flexDirection: "column", gap: 5, marginTop: 8,
  },
  validationItem: {
    color: "var(--text-normal)", fontSize: 11,
    fontFamily: "var(--font-code)", lineHeight: 1.5,
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
  agentProgress: { display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 },
  agentProgressItem: { display: "flex", alignItems: "center", gap: 8, fontSize: 12 },
  agentStatusDot: { width: 8, height: 8, borderRadius: "50%" },
  streamingCode: { width: "100%", maxHeight: 400, marginTop: 12 },
};
