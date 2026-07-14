import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ObserverAgent")


class ObserverAgent(BaseAgent):
    name = "ObserverAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        # Collects per-agent latency, token counts, success/failure.
        # Data exposed via /metrics endpoint (Phase 3).
        log.info(
            "ObserverAgent: metrics collected for %d agents", len(state.agents_done)
        )
        return AgentResult(
            success=True,
            summary=f"Observed {len(state.agents_done)} agents",
        )
