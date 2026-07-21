import { useMemo, useRef } from "react";

export const countLines = (text) => (text ? text.split("\n").length : 1);

/**
 * Code pane with a line-number gutter.
 *
 * The gutter is a separate scroll container kept in sync with the code on
 * every scroll event, so numbers never drift from their lines. Alignment
 * depends on both elements sharing --code-size / --code-line and the same
 * vertical padding — see index.css. Lines never wrap (wrap="off" /
 * white-space: pre), which is what keeps one number to one visual row.
 */
export default function CodeSurface({ value, onChange, placeholder }) {
  const gutterRef = useRef(null);
  const numbers = useMemo(() => {
    const total = countLines(value);
    let out = "";
    for (let i = 1; i <= total; i++) out += (i > 1 ? "\n" : "") + i;
    return out;
  }, [value]);

  const syncScroll = (e) => {
    if (gutterRef.current) gutterRef.current.scrollTop = e.currentTarget.scrollTop;
  };

  return (
    <div className="code-wrap">
      <div className="gutter" ref={gutterRef} aria-hidden="true">
        <pre>{numbers}</pre>
      </div>

      {onChange ? (
        <textarea
          className="code"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onScroll={syncScroll}
          placeholder={placeholder}
          wrap="off"
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
        />
      ) : (
        <pre className="code" onScroll={syncScroll} tabIndex={0}>{value}</pre>
      )}
    </div>
  );
}
