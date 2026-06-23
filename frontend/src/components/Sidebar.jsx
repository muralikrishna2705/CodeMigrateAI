import AgentLog from "./AgentLog.jsx";

const LANGUAGES = [
  { id: "java",       name: "Java",       versions: ["7","8","11","17","21"] },
  { id: "python",     name: "Python",     versions: ["2.7","3.8","3.10","3.12"] },
  { id: "javascript", name: "JavaScript", versions: ["ES5","ES6","ES2020","ES2022"] },
  { id: "typescript", name: "TypeScript", versions: ["3.x","4.x","5.x"] },
  { id: "csharp",     name: "C#",         versions: ["6","8","10","12"] },
  { id: "go",         name: "Go",         versions: ["1.18","1.20","1.22"] },
  { id: "kotlin",     name: "Kotlin",     versions: ["1.7","1.9","2.0"] },
  { id: "rust",       name: "Rust",       versions: ["1.70","1.80"] },
  { id: "cpp",        name: "C++",        versions: ["14","17","20","23"] },
];

export default function Sidebar({
  srcLang, setSrcLang, srcVer, setSrcVer,
  tgtLang, setTgtLang, tgtVer, setTgtVer,
  pipelineMode, setPipelineMode,
  loading, ollamaStatus, apiError, result, agentProgress, onRun,
}) {
  const srcDef = LANGUAGES.find(l => l.id === srcLang) || LANGUAGES[0];
  const tgtDef = LANGUAGES.find(l => l.id === tgtLang) || LANGUAGES[0];

  const handleSrcLang = (id) => {
    setSrcLang(id);
    setSrcVer(LANGUAGES.find(l => l.id === id)?.versions[0] || "");
  };
  const handleTgtLang = (id) => {
    setTgtLang(id);
    const vs = LANGUAGES.find(l => l.id === id)?.versions || [];
    setTgtVer(vs[vs.length - 1] || "");
  };

  const isConversion  = srcLang !== tgtLang;
  const canRun        = ollamaStatus === "ok" && !loading;

  return (
    <aside style={st.aside}>

      {/* ── Migration type badge ── */}
      <div style={st.badgeWrap}>
        <span style={{ ...st.badge, ...(isConversion ? st.badgeConvert : st.badgeUpgrade) }}>
          {isConversion ? "⇄ Language Conversion" : "↑ Version Upgrade"}
        </span>
      </div>

      {/* ── Source ── */}
      <SectionLabel text="Source" />
      <SelectRow
        label="Language"
        value={srcLang}
        onChange={handleSrcLang}
        options={LANGUAGES.map(l => ({ value: l.id, label: l.name }))}
      />
      <SelectRow
        label="Version"
        value={srcVer}
        onChange={setSrcVer}
        options={srcDef.versions.map(v => ({ value: v, label: v }))}
      />

      <div style={st.arrow}>↓</div>

      {/* ── Target ── */}
      <SectionLabel text="Target" />
      <SelectRow
        label="Language"
        value={tgtLang}
        onChange={handleTgtLang}
        options={LANGUAGES.map(l => ({ value: l.id, label: l.name }))}
      />
      <SelectRow
        label="Version"
        value={tgtVer}
        onChange={setTgtVer}
        options={tgtDef.versions.map(v => ({ value: v, label: v }))}
      />

      {/* ── Pipeline Mode ── */}
      <SectionLabel text="Mode" />
      <SelectRow
        label="Pipeline"
        value={pipelineMode}
        onChange={setPipelineMode}
        options={[
          { value: "fast", label: "Fast (1 LLM call)" },
          { value: "deep", label: "Deep (3 LLM calls)" },
          { value: "validated", label: "Validated (+syntax)" },
        ]}
      />

      {/* ── Run button ── */}
      <button
        style={{ ...st.runBtn, ...(canRun ? {} : st.runBtnDisabled) }}
        onClick={onRun}
        disabled={!canRun}
      >
        {loading
          ? <><Spinner /> Migrating…</>
          : <>"⟳  Run Migration"</>
        }
      </button>

      {/* ── Warnings / Errors ── */}
      {ollamaStatus === "down" && !loading && (
        <div style={st.warnBox}>
          ⚠ Ollama offline. Run:<br />
          <code style={st.code}>ollama serve</code><br />
          <code style={st.code}>ollama pull deepseek-coder-v2</code>
        </div>
      )}
      {apiError && (
        <div style={st.errBox}>⚠ {apiError}</div>
      )}

      {/* ── Agent pipeline log ── */}
      {(result || agentProgress.length > 0) && (
        <div style={st.logWrap}>
          <SectionLabel text="Agent Pipeline" />
          {agentProgress.length > 0 && !result ? (
            <div style={st.streamingProgress}>
              {agentProgress.map((a, i) => (
                <div key={i} style={st.agentProgressItem}>
                  <span style={{
                    ...st.agentStatusDot,
                    background: a.status === "complete" ? "var(--green)" : "var(--accent)",
                  }} />
                  <span style={{ fontSize: 11, color: "var(--text-normal)" }}>
                    {a.agent} {a.status === "complete" ? "✓" : "⟳"}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <AgentLog reports={result.reports} />
          )}
        </div>
      )}
    </aside>
  );
}

/* ── Small reusable components ─────────────────────────────────── */

function SectionLabel({ text }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, textTransform: "uppercase",
      letterSpacing: "1.2px", color: "var(--text-dim)", marginBottom: 6,
    }}>
      {text}
    </div>
  );
}

function SelectRow({ label, value, onChange, options }) {
  return (
    <div style={st.selectRow}>
      <label style={st.selectLabel}>{label}</label>
      <select
        style={st.select}
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

function Spinner() {
  return (
    <span style={{
      display: "inline-block", width: 13, height: 13,
      border: "2px solid rgba(0,0,0,.25)", borderTopColor: "#000",
      borderRadius: "50%", animation: "spin .65s linear infinite",
    }} />
  );
}

/* ── Styles ───────────────────────────────────────────────────── */
const st = {
  aside: {
    width: "var(--sidebar)", minWidth: "var(--sidebar)",
    background: "var(--bg-base)", borderRight: "1px solid var(--border)",
    display: "flex", flexDirection: "column", gap: 8,
    padding: "16px 14px", overflowY: "auto", overflowX: "hidden",
  },
  badgeWrap: { marginBottom: 4 },
  badge: {
    display: "inline-flex", alignItems: "center", gap: 5,
    padding: "4px 11px", borderRadius: "var(--radius)",
    fontSize: 11, fontWeight: 700, letterSpacing: ".5px",
    border: "1px solid",
  },
  badgeUpgrade: {
    background: "rgba(56,189,248,.1)", borderColor: "rgba(56,189,248,.35)",
    color: "var(--blue)",
  },
  badgeConvert: {
    background: "rgba(167,139,250,.1)", borderColor: "rgba(167,139,250,.35)",
    color: "var(--purple)",
  },
  selectRow: { display: "flex", alignItems: "center", gap: 8, marginBottom: 2 },
  selectLabel: { fontSize: 11, color: "var(--text-muted)", minWidth: 52 },
  select: {
    flex: 1, padding: "5px 24px 5px 8px",
    background: "var(--bg-raised)", border: "1px solid var(--border-mid)",
    borderRadius: "var(--radius)", color: "var(--text-bright)",
    fontSize: 12, fontFamily: "var(--font-ui)", cursor: "pointer",
    appearance: "none",
    backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'%3E%3Cpath fill='%23506070' d='M5 7L0 2h10z'/%3E%3C/svg%3E\")",
    backgroundRepeat: "no-repeat", backgroundPosition: "right 7px center",
  },
  arrow: { textAlign: "center", color: "var(--accent)", fontSize: 18, margin: "2px 0" },
  runBtn: {
    padding: "11px 0", background: "var(--accent)", border: "none",
    borderRadius: "var(--radius)", color: "#000",
    fontSize: 13, fontWeight: 700, fontFamily: "var(--font-ui)",
    cursor: "pointer", display: "flex", alignItems: "center",
    justifyContent: "center", gap: 8, marginTop: 4,
    transition: "all .2s",
  },
  runBtnDisabled: {
    background: "var(--bg-overlay)", color: "var(--text-dim)", cursor: "not-allowed",
  },
  warnBox: {
    padding: "10px 12px", background: "rgba(251,191,36,.07)",
    border: "1px solid rgba(251,191,36,.25)", borderRadius: "var(--radius)",
    fontSize: 11, color: "var(--amber)", lineHeight: 1.6,
  },
  errBox: {
    padding: "10px 12px", background: "rgba(248,113,113,.07)",
    border: "1px solid rgba(248,113,113,.25)", borderRadius: "var(--radius)",
    fontSize: 11, color: "var(--red)", lineHeight: 1.5,
  },
  code: {
    display: "inline-block", marginTop: 3,
    background: "rgba(251,191,36,.12)", padding: "1px 5px",
    borderRadius: 3, fontFamily: "var(--font-code)", fontSize: 10,
  },
  logWrap: { display: "flex", flexDirection: "column", gap: 6, marginTop: 4 },
  streamingProgress: { display: "flex", flexDirection: "column", gap: 4 },
  agentProgressItem: { display: "flex", alignItems: "center", gap: 6, fontSize: 11 },
  agentStatusDot: { width: 8, height: 8, borderRadius: "50%" },
};