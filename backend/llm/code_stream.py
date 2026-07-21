"""Incremental unwrap of the streamed migrator JSON into clean code tokens.

The MigratorAgent asks the model for a single JSON object
``{"plan_summary": ..., "migrated_code": ...}`` and keeps that JSON as the
internal contract between agents. Streaming the *raw* tokens to the browser,
however, shows the user the JSON scaffolding and escape sequences
(``{"plan_summary": "…", "migrated_code": "def f():\\n    ...``) instead of the
target-language code they asked for.

:class:`MigratedCodeStreamer` sits between the raw token stream and the SSE
callback: it walks the accumulating buffer, locates the ``migrated_code`` string
value, and emits only its *decoded* contents as they arrive. So the live stream
carries clean code while the full JSON is still assembled internally for
parsing. It is deliberately tolerant of tokens that split anywhere — mid-key,
mid-escape, or mid-``\\uXXXX`` — because network/LLM token boundaries are
arbitrary.

If the response never forms the expected JSON shape (rare, since the migrator
calls with ``fmt="json"``), nothing is emitted live and the authoritative code
still reaches the client via the terminal ``complete`` event.
"""

import re

# Matches the ``migrated_code`` object key through the opening quote of its
# string value. Requiring the ``:`` and opening ``"`` (not a bare substring)
# guards against the literal text "migrated_code" appearing inside an earlier
# value such as plan_summary.
_VALUE_START_RE = re.compile(r'"migrated_code"\s*:\s*"')

# JSON single-character escape sequences (the char after the backslash -> its
# decoded value). Anything not listed is emitted literally, which keeps the
# stream lenient against a model that over-escapes.
_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
}


class MigratedCodeStreamer:
    """Feed raw JSON tokens, get back decoded ``migrated_code`` deltas."""

    def __init__(self) -> None:
        self._buf = ""
        self._pos = 0
        self._in_value = False
        self._done = False
        self._decoded: list[str] = []

    def feed(self, delta: str) -> str:
        """Append ``delta`` to the buffer and return any newly decoded code.

        Returns an empty string until the ``migrated_code`` value has started,
        and after its closing quote. The returned text is already JSON-decoded
        (escapes resolved), ready to show verbatim.
        """
        if self._done or not delta:
            return ""
        self._buf += delta

        if not self._in_value:
            match = _VALUE_START_RE.search(self._buf)
            if not match:
                return ""
            self._in_value = True
            self._pos = match.end()

        chunk = self._drain_value()
        if chunk:
            self._decoded.append(chunk)
        return chunk

    def value(self) -> str:
        """Everything decoded from ``migrated_code`` so far.

        The salvage path uses this when the completed response won't parse: the
        streamer has been tracking the code field character by character all
        along, so its view survives a response that was truncated before its
        closing brace — which is exactly when parsing fails.
        """
        return "".join(self._decoded)

    def _drain_value(self) -> str:
        buf = self._buf
        out: list[str] = []
        pos = self._pos
        n = len(buf)

        while pos < n:
            ch = buf[pos]
            if ch == "\\":
                # Need at least the escape indicator; wait for more if it (or a
                # \uXXXX payload) has not fully arrived, without consuming the
                # backslash so we can resume cleanly on the next feed.
                if pos + 1 >= n:
                    break
                esc = buf[pos + 1]
                if esc == "u":
                    if pos + 6 > n:
                        break
                    hex4 = buf[pos + 2 : pos + 6]
                    try:
                        out.append(chr(int(hex4, 16)))
                    except ValueError:
                        # Malformed unicode escape: pass through literally
                        # rather than dropping code the user cares about.
                        out.append(buf[pos : pos + 6])
                    pos += 6
                else:
                    out.append(_ESCAPES.get(esc, esc))
                    pos += 2
            elif ch == '"':
                # Unescaped quote terminates the string value.
                self._done = True
                pos += 1
                break
            else:
                out.append(ch)
                pos += 1

        self._pos = pos
        return "".join(out)
