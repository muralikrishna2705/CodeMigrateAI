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

      while (true) {
        const { done, value } = await readerRef.current.read();
        if (done) break;

        const lines = decoder.decode(value).split("\n\n");
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const event = JSON.parse(line.slice(6));

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
              return event.result;
            case "error":
              setError(event.message);
              setLoading(false);
              throw new Error(event.message);
          }
        }
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

  return { result, streamBuffer, agentProgress, loading, error, runMigration, cancel };
}