import CodeSurface, { countLines } from "./CodeSurface.jsx";

export default function Editor({ code, onChange, language, version }) {
  const lines = countLines(code);

  return (
    <section className="pane" aria-label="Source code">
      <div className="pane-head">
        <span className="pane-title label">{language.toUpperCase()} {version} · Source</span>
        <div className="pane-actions">
          <span className="pane-meta">{lines} {lines === 1 ? "line" : "lines"}</span>
          <button className="btn btn--subtle" onClick={() => onChange("")} disabled={!code}>
            Clear
          </button>
        </div>
      </div>

      <CodeSurface
        value={code}
        onChange={onChange}
        placeholder={`Paste your ${language} code here…`}
      />
    </section>
  );
}
