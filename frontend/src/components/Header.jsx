export default function Header({ ollamaStatus, examples, onLoadExample }) {
  const statusColors = {
    ok:       { color: "#22d3a0", dot: "#22d3a0", label: "DeepSeek-Coder-V2 Ready" },
    down:     { color: "#f87171", dot: "#f87171", label: "Ollama Offline"          },
    checking: { color: "#506070", dot: "#506070", label: "Checking…"               },
  };
  const s = statusColors[ollamaStatus] || statusColors.checking;

  return (
    <header style={st.header}>
      {/* Brand */}
      <div style={st.brand}>
        <span style={st.logoGlyph}>⟳</span>
        <div>
          <span style={st.logoText}>
            Code<span style={{ color: "var(--accent)", fontWeight: 800 }}>Migrate</span>AI
          </span>
          <span style={st.logoSub}>AI-Driven Code Migration Platform</span>
        </div>
      </div>

      {/* Quick-load examples */}
      <div style={st.center}>
        <span style={st.exLabel}>Try:</span>
        {examples.map((name) => (
          <button key={name} style={st.exBtn}
            onClick={() => onLoadExample(name)}
            onMouseEnter={e => Object.assign(e.target.style, { borderColor: "var(--accent)", color: "var(--accent)" })}
            onMouseLeave={e => Object.assign(e.target.style, { borderColor: "var(--border-mid)", color: "var(--text-muted)" })}
          >
            {name}
          </button>
        ))}
      </div>

      {/* Status pill */}
      <div style={{ ...st.pill, borderColor: s.color + "44", color: s.color }}>
        <span style={{ ...st.dot, background: s.dot,
          boxShadow: ollamaStatus === "ok" ? `0 0 6px ${s.dot}` : "none",
          animation:  ollamaStatus === "ok" ? "pulse 2s infinite" : "none",
        }} />
        {s.label}
        <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}`}</style>
      </div>
    </header>
  );
}

const st = {
  header: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    height: "var(--header)", padding: "0 20px",
    background: "var(--bg-base)", borderBottom: "1px solid var(--border)",
    flexShrink: 0,
  },
  brand: { display: "flex", alignItems: "center", gap: 12 },
  logoGlyph: {
    fontSize: 24, color: "var(--accent)",
    display: "inline-block", animation: "spin 9s linear infinite",
  },
  logoText: {
    display: "block", fontSize: 17, fontWeight: 600,
    fontFamily: "var(--font-ui)", letterSpacing: "-0.4px",
  },
  logoSub: {
    display: "block", fontSize: 10, color: "var(--text-dim)",
    textTransform: "uppercase", letterSpacing: "1px",
  },
  center: { display: "flex", alignItems: "center", gap: 8 },
  exLabel: { fontSize: 11, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "1px" },
  exBtn: {
    padding: "4px 12px", background: "var(--bg-raised)",
    border: "1px solid var(--border-mid)", borderRadius: "var(--radius)",
    color: "var(--text-muted)", fontSize: 11,
    fontFamily: "var(--font-ui)", cursor: "pointer", transition: "all .15s",
  },
  pill: {
    display: "flex", alignItems: "center", gap: 7,
    padding: "4px 12px", borderRadius: 20, fontSize: 11,
    border: "1px solid", background: "var(--bg-base)",
  },
  dot: { width: 7, height: 7, borderRadius: "50%" },
};