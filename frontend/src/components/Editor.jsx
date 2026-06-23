export default function Editor({ code, onChange, language, version }) {
  const lines = code.split("\n").length;

  return (
    <div style={st.panel}>
      <div style={st.header}>
        <span style={st.label}>
          {language.toUpperCase()} {version} — Source
        </span>
        <div style={st.actions}>
          <span style={st.lineCount}>{lines} lines</span>
          <button
            style={st.btn}
            onClick={() => onChange("")}
            onMouseEnter={e => e.target.style.color = "var(--accent)"}
            onMouseLeave={e => e.target.style.color = "var(--text-muted)"}
          >
            Clear
          </button>
        </div>
      </div>
      <textarea
        style={st.textarea}
        value={code}
        onChange={e => onChange(e.target.value)}
        placeholder={`Paste your ${language} code here…`}
        spellCheck={false}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="off"
      />
    </div>
  );
}

const st = {
  panel: { display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" },
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
  lineCount: { fontSize: 10, color: "var(--text-dim)", fontFamily: "var(--font-code)" },
  btn: {
    padding: "2px 9px", background: "var(--bg-overlay)",
    border: "1px solid var(--border-mid)", borderRadius: 4,
    color: "var(--text-muted)", fontSize: 10,
    fontFamily: "var(--font-ui)", cursor: "pointer", transition: "color .15s",
  },
  textarea: {
    flex: 1, background: "var(--bg-root)", border: "none",
    color: "var(--text-bright)", fontFamily: "var(--font-code)",
    fontSize: 12.5, lineHeight: 1.75, padding: "14px 18px",
    resize: "none", outline: "none", overflow: "auto", tabSize: 4,
  },
};