import { useState, useCallback, useRef } from "react";

export function useMigrationStream(apiBase) {
  const [result, setResult] = useState(null);
  const [streamBuffer, setStreamBuffer] = useState("");
  const [agentProgress, setAgentProgress] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const readerRef = useRef(null);
  const abortControllerRef = useRef(null);

  const runMigration = useCallback(async (params) => {
    setLoading(true);
    setError(null);
    setStreamBuffer("");
    setAgentProgress([]);
    setResult(null);

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${apiBase}/migrate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
        signal: abortControllerRef.current.signal,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      readerRef.current = response.body.getReader();
      const decoder = new TextDecoder();

      // Network chunk boundaries are arbitrary and do NOT align with SSE event
      // boundaries: a single read() may deliver half an event, or several
      // events plus a partial one. Accumulate raw text and only consume whole
      // `...\n\n`-terminated frames, carrying any trailing partial frame over
      // to the next read. `{ stream: true }` likewise keeps multi-byte UTF-8
      // sequences intact when they straddle a chunk boundary.
      let buffer = "";

      const handleEvent = (event) => {
        switch (event.type) {
          case "agent_start":
            setAgentProgress((p) => [
              ...p,
              { agent: event.agent, status: "running", message: event.message },
            ]);
            break;
          case "agent_complete":
            setAgentProgress((p) =>
              p.map((a) =>
                a.agent === event.agent
                  ? { ...a, status: "complete", ...event }
                  : a
              )
            );
            break;
          case "token":
            setStreamBuffer((b) => b + event.content);
            break;
          case "complete":
            setResult(event.result);
            setLoading(false);
            return { terminal: "complete", value: event.result };
          case "error":
            setError(event.message);
            setLoading(false);
            throw new Error(event.message);
        }
        return null;
      };

      // Parse every complete frame currently in `buffer`, leaving the trailing
      // partial (if any) behind. Returns a terminal result to unwind on, or null.
      const drainBuffer = () => {
        let sepIndex;
        while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);
          // An SSE frame may carry multiple `data:` lines; join their payloads.
          const dataLines = frame
            .split("\n")
            .filter((l) => l.startsWith("data: "))
            .map((l) => l.slice(6));
          if (dataLines.length === 0) continue;
          const event = JSON.parse(dataLines.join("\n"));
          const terminal = handleEvent(event);
          if (terminal) return terminal;
        }
        return null;
      };

      while (true) {
        const { done, value } = await readerRef.current.read();
        if (done) {
          // Flush any final buffered frame the stream ended without a trailing
          // blank line (defensive; well-formed SSE always terminates a frame).
          buffer += decoder.decode();
          const terminal = drainBuffer();
          if (terminal) return terminal.value;
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const terminal = drainBuffer();
        if (terminal) return terminal.value;
      }
    } catch (err) {
      if (err.name === "AbortError") {
        setLoading(false);
        return;
      }
      setError(err.message);
      setLoading(false);
      throw err;
    } finally {
      readerRef.current = null;
      abortControllerRef.current = null;
    }
  }, [apiBase]);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (readerRef.current) {
      readerRef.current.cancel();
    }
    setLoading(false);
  }, []);

  return { result, streamBuffer, agentProgress, loading, error, setError, runMigration, cancel };
}