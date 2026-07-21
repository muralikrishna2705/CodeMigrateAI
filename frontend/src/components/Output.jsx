import { useState } from "react";
import CodeSurface, { countLines } from "./CodeSurface.jsx";

export default function Output({ result, streamBuffer, loading, language, version, agentProgress }) {
  const [copied, setCopied] = useState(false);

  const displayCode = result?.migrated_code || streamBuffer || "";

  const handleCopy = () => {
    if (!displayCode) return;
    navigator.clipboard.writeText(displayCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const title = `${language.toUpperCase()} ${version} · Output`;

  // ── Running, nothing generated yet ─────────────────────────────────────────
  if (loading && !displayCode) {
    const live = agentProgress || [];
    return (
      <section className="pane" aria-label="Migration output" aria-busy="true">
        <div className="pane-head">
          <span className="pane-title label">Running agent pipeline…</span>
        </div>
        <div className="state">
          {live.length > 0 ? (
            <div className="live-list">
              {live.map((a, i) => {
                const done = a.status === "complete";
                return (
                  <div key={i} className={"live-item" + (done ? " live-item--done" : "")}>
                    <span
                      className={"dot" + (done ? "" : " dot--live")}
                      style={{ background: done ? "var(--success)" : "var(--accent)" }}
                    />
                    {a.agent} {done ? "✓" : "⟳"}
                    {a.message && <span className="live-msg">{a.message}</span>}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="dots">
              <span style={{ animationDelay: "0s" }} />
              <span style={{ animationDelay: ".2s" }} />
              <span style={{ animationDelay: ".4s" }} />
            </div>
          )}
          <div className="sweep" />
          <p className="state-hint">Analyzing structure, then generating {language} {version}</p>
        </div>
      </section>
    );
  }

  // ── Empty ──────────────────────────────────────────────────────────────────
  if (!displayCode) {
    return (
      <section className="pane" aria-label="Migration output">
        <div className="pane-head">
          <span className="pane-title label">{title}</span>
        </div>
        <div className="state">
          <div className="state-mark" aria-hidden="true">⤓</div>
          <p className="state-title">Migrated code appears here</p>
          <p className="state-hint">Set your target on the left, then run the migration</p>
        </div>
      </section>
    );
  }

  // ── Result (or live stream) ────────────────────────────────────────────────
  const planSummary = result?.inline_plan || "";
  const validation  = getValidationState(result?.validation_result);
  const lines       = countLines(displayCode);

  return (
    <section className="pane" aria-label="Migration output" aria-busy={loading || undefined}>
      <div className="pane-head">
        <span className="pane-title label">{title}</span>
        <div className="pane-actions">
          {loading && <span className="tag tag--quiet">⟳ Streaming</span>}
          {result && (
            <span className={"tag " + (result.success ? "tag--success" : "tag--warn")}>
              {result.success ? "✓ Success" : "⚠ Partial"}
            </span>
          )}
          {validation && <span className={"tag " + validation.tone}>{validation.label}</span>}
          <span className="pane-meta">{lines} {lines === 1 ? "line" : "lines"}</span>
          <button className="btn btn--subtle" onClick={handleCopy} disabled={!displayCode}>
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
      </div>

      {planSummary && (
        <div className="band">
          <span className="label">Plan</span>
          <span>{planSummary}</span>
        </div>
      )}

      {validation?.items.length > 0 && (
        <div className="band">
          <details>
            <summary>{validation.items.length} validation {validation.items.length === 1 ? "note" : "notes"}</summary>
            <ul className="band-list">
              {validation.items.map((item, i) => <li key={i}>{formatIssue(item)}</li>)}
            </ul>
          </details>
        </div>
      )}

      <CodeSurface value={displayCode} />
    </section>
  );
}

function getValidationState(validation) {
  if (!validation) return null;

  const valid = Boolean(validation.valid ?? validation.syntax_valid);
  const syntaxErrors = validation.syntax_errors || {};
  const errors = validation.errors || syntaxErrors.errors || [];
  const warnings = validation.warnings || syntaxErrors.warnings || [];
  const items = [...errors, ...warnings];

  if (!valid)             return { label: "Validation failed",   tone: "tag--danger",  items };
  if (warnings.length)    return { label: "Validation warnings", tone: "tag--warn",    items };
  return                         { label: "Validation OK",       tone: "tag--success", items: [] };
}

function formatIssue(issue) {
  const location = issue.line ? `${issue.line}:${issue.column || 0} ` : "";
  return `${location}${issue.message || String(issue)}`;
}
