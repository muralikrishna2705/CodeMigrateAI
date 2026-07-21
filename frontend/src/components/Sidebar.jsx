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

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || "");
const RUN_KEY = IS_MAC ? "⌘ ↵" : "Ctrl ↵";

export default function Sidebar({
  srcLang, setSrcLang, srcVer, setSrcVer,
  tgtLang, setTgtLang, tgtVer, setTgtVer,
  loading, ollamaStatus, ollamaModel, apiError, result, agentProgress,
  onRun, onCancel,
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

  // Reversing the route is a one-click operation in a migration tool.
  const swap = () => {
    setSrcLang(tgtLang); setSrcVer(tgtVer);
    setTgtLang(srcLang); setTgtVer(srcVer);
  };

  const isConversion = srcLang !== tgtLang;
  const canRun       = ollamaStatus === "ok" && !loading;

  const srcVerIndex = srcDef.versions.indexOf(srcVer);
  const tgtVerIndex = tgtDef.versions.indexOf(tgtVer);
  const isDowngrade = !isConversion && srcVerIndex >= 0 && tgtVerIndex >= 0 && tgtVerIndex < srcVerIndex;

  return (
    <aside className="sidebar">
      <span className={"tag " + (isConversion ? "tag--violet" : "tag--info")} style={{ alignSelf: "flex-start" }}>
        {isConversion ? "⇄ Language Conversion" : "↑ Version Upgrade"}
      </span>

      {/* ── Route: source → target ── */}
      <div className="route">
        <div className="route-card">
          <span className="label" id="src-label">From</span>
          <div className="route-selects">
            <select
              className="select select--lang" value={srcLang}
              onChange={e => handleSrcLang(e.target.value)}
              aria-label="Source language"
            >
              {LANGUAGES.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
            </select>
            <select
              className="select select--ver" value={srcVer}
              onChange={e => setSrcVer(e.target.value)}
              aria-label="Source version"
            >
              {srcDef.versions.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>

        <div className="route-link">
          <button className="btn btn--icon" onClick={swap} title="Swap source and target" aria-label="Swap source and target">
            ⇅
          </button>
        </div>

        <div className="route-card">
          <span className="label">To</span>
          <div className="route-selects">
            <select
              className="select select--lang" value={tgtLang}
              onChange={e => handleTgtLang(e.target.value)}
              aria-label="Target language"
            >
              {LANGUAGES.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
            </select>
            <select
              className="select select--ver" value={tgtVer}
              onChange={e => setTgtVer(e.target.value)}
              aria-label="Target version"
            >
              {tgtDef.versions.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* ── Run / Cancel ── */}
      <div className="stack">
        {loading ? (
          <>
            <button className="btn btn--primary" disabled aria-busy="true">
              <Spinner /> Migrating…
            </button>
            <button className="btn btn--stop" onClick={onCancel}>Cancel</button>
            <p className="run-hint">or press <kbd>Esc</kbd></p>
          </>
        ) : (
          <>
            <button className="btn btn--primary" onClick={onRun} disabled={!canRun}>
              Run Migration <kbd>{RUN_KEY}</kbd>
            </button>
            {!canRun && <p className="run-hint">Waiting for the LLM backend</p>}
          </>
        )}
      </div>

      {/* ── Alerts ── */}
      {ollamaStatus === "down" && !loading && (
        <div className="alert alert--warn" role="status">
          Backend or Ollama unreachable. Start it with:<br />
          <code>ollama serve</code><br />
          <code>{ollamaModel || "the configured model"}</code> is pulled automatically
          when the backend starts.
        </div>
      )}

      {apiError && (
        <div className="alert alert--danger" role="alert">{apiError}</div>
      )}

      {isDowngrade && (
        <div className="alert alert--warn" role="status">
          Downgrading {srcDef.name} {srcVer} → {tgtVer} may introduce breaking
          changes. Check backward compatibility.
        </div>
      )}

      {/* ── Agent pipeline ── */}
      {(result || agentProgress.length > 0) && (
        <div className="pipeline">
          <span className="label">Agent Pipeline</span>
          {result ? (
            <AgentLog reports={result.reports} />
          ) : (
            <div className="live-list">
              {agentProgress.map((a, i) => {
                const done = a.status === "complete";
                return (
                  <div key={i} className={"live-item" + (done ? " live-item--done" : "")}>
                    <span
                      className={"dot" + (done ? "" : " dot--live")}
                      style={{ background: done ? "var(--success)" : "var(--accent)" }}
                    />
                    {a.agent} {done ? "✓" : "⟳"}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

function Spinner() {
  return (
    <span
      aria-hidden="true"
      style={{
        display: "inline-block", width: 13, height: 13,
        border: "2px solid rgba(0,0,0,.3)", borderTopColor: "currentColor",
        borderRadius: "50%", animation: "spin .65s linear infinite",
      }}
    />
  );
}
