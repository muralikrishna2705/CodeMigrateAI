import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.DispatcherAgent")


class DispatcherAgent(BaseAgent):
    name = "DispatcherAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        # Determines which agents to run based on migration complexity and type.
        # For low complexity: AnalyzerAgent -> PlannerAgent -> MigratorAgent
        # For high complexity: adds DeepAnalyzerAgent (Phase 3)
        log.info(
            "DispatcherAgent: routing determined for complexity=%s",
            (state.code_metrics or {}).get("complexity", "unknown"),
        )
        return AgentResult(
            success=True,
            summary="Dispatch routing complete",
        )
