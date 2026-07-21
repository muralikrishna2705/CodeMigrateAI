export default function Header({ ollamaStatus, ollamaModel, examples, onLoadExample }) {
  // The pill reports machine state, not marketing. When the LLM is up, the
  // most useful thing to show is which model is actually answering.
  const status = {
    ok: {
      color: "var(--accent)",
      border: "var(--accent-line)",
      text: ollamaModel || "LLM ready",
      mono: Boolean(ollamaModel),
      title: "LLM backend connected",
    },
    down: {
      color: "var(--danger)",
      border: "var(--danger-line)",
      text: "LLM offline",
      mono: false,
      title: "Backend or Ollama unreachable",
    },
    checking: {
      color: "var(--text-tertiary)",
      border: "var(--border)",
      text: "Connecting…",
      mono: false,
      title: "Checking backend health",
    },
  }[ollamaStatus] || {
    color: "var(--text-tertiary)", border: "var(--border)",
    text: "Connecting…", mono: false, title: "Checking backend health",
  };

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true"><span>⟳</span></div>
        <div>
          <span className="brand-name">Code<b>Migrate</b>AI</span>
          <span className="brand-sub">LLM-Powered Migration Platform</span>
        </div>
      </div>

      <div className="examples">
        <span className="label">Try</span>
        {examples.map((name) => (
          <button
            key={name}
            className="btn btn--chip"
            onClick={() => onLoadExample(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <div
        className="status"
        style={{ color: status.color, borderColor: status.border }}
        title={status.title}
        aria-live="polite"
      >
        <span
          className={"dot" + (ollamaStatus === "ok" ? " dot--live" : "")}
          style={{
            background: status.color,
            boxShadow: ollamaStatus === "ok" ? `0 0 7px ${status.color}` : "none",
          }}
        />
        <span className={status.mono ? "status-model" : undefined}>{status.text}</span>
      </div>
    </header>
  );
}
