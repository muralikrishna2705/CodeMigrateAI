"""Circuit breaker for graph agents.

Formerly a linear-prelude ``RecoveryAgent`` that ran once per request and whose
results nothing consumed. The breaker is now consulted directly by the graph
node factory (``graph.nodes._make_node``): a node checks ``is_circuit_open``
before constructing its agent and calls ``record_failure`` when the agent
records an error, so a repeatedly failing agent (e.g. LLM timeouts) is skipped
for a cooldown window instead of being retried on every request.

State is process-local (``_circuit_state``); durable Redis-backed storage is a
deferred follow-up (GAP17).
"""

import logging
import time

log = logging.getLogger("CodeMigrateAI.CircuitBreaker")

# Circuit breaker state, keyed by agent name: {agent: {failures, open_until}}.
_circuit_state: dict[str, dict] = {}

# Consecutive failures before the circuit opens, and how long it stays open.
FAILURE_THRESHOLD = 3
COOLDOWN_SEC = 60


def record_failure(agent_name: str) -> None:
    """Record one failure for ``agent_name``; open the circuit at the threshold."""
    state = _circuit_state.setdefault(agent_name, {"failures": 0, "open_until": 0})
    state["failures"] += 1
    if state["failures"] >= FAILURE_THRESHOLD:
        state["open_until"] = time.time() + COOLDOWN_SEC
        log.warning(
            "Circuit opened for %s after %d failures (cooldown %ds)",
            agent_name,
            state["failures"],
            COOLDOWN_SEC,
        )


def is_circuit_open(agent_name: str) -> bool:
    """Return True if the circuit for ``agent_name`` is currently open.

    A cooldown that has elapsed closes the circuit and resets the failure count,
    so the agent gets a clean retry on the next request.
    """
    state = _circuit_state.get(agent_name)
    if not state:
        return False
    if state.get("open_until", 0) > time.time():
        return True
    if state["open_until"] > 0:  # cooldown expired -> close + reset
        state["failures"] = 0
        state["open_until"] = 0
    return False


def reset() -> None:
    """Clear all circuit state (used by tests)."""
    _circuit_state.clear()


def snapshot() -> dict[str, dict]:
    """Return a shallow copy of the circuit state (for metrics/observability)."""
    return {
        name: {"failures": s["failures"], "open_until": s.get("open_until", 0)}
        for name, s in _circuit_state.items()
    }
