import logging
import time

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RecoveryAgent")

# Circuit breaker state, keyed by agent name.
_circuit_state: dict[str, dict] = {}


class RecoveryAgent(BaseAgent):
    name = "RecoveryAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        errors = state.errors
        recovery_actions = []

        for error_text in errors:
            agent_name = (
                error_text.split("]")[0].lstrip("[")
                if "]" in error_text
                else "unknown"
            )
            self._record_failure(agent_name)

            if self._is_circuit_open(agent_name):
                recovery_actions.append(f"Circuit open for {agent_name}, skipping")
                log.warning("Circuit breaker open for %s", agent_name)
            else:
                recovery_actions.append(f"Will retry {agent_name}")

        return AgentResult(
            success=True,
            summary=f"Recovery: {len(errors)} errors, {len(recovery_actions)} actions",
            details={
                "error_count": len(errors),
                "recovery_actions": recovery_actions,
                "circuit_states": {
                    k: {"failures": v["failures"], "open_until": v.get("open_until", 0)}
                    for k, v in _circuit_state.items()
                },
            },
        )

    def _record_failure(self, agent_name: str):
        if agent_name not in _circuit_state:
            _circuit_state[agent_name] = {"failures": 0, "open_until": 0}
        state = _circuit_state[agent_name]
        state["failures"] += 1
        if state["failures"] >= 3:
            state["open_until"] = time.time() + 60  # 60-second cooldown

    def _is_circuit_open(self, agent_name: str) -> bool:
        state = _circuit_state.get(agent_name)
        if not state:
            return False
        if state.get("open_until", 0) > time.time():
            return True
        # Cooldown expired -> close the circuit and reset the failure count.
        if state["open_until"] > 0:
            state["failures"] = 0
            state["open_until"] = 0
        return False
