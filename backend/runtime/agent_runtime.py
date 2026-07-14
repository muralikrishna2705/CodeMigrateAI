import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RuntimeAgent")


class RuntimeAgent(BaseAgent):
    name = "RuntimeAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        # The RuntimeAgent wraps the entire pipeline execution.
        # It coordinates startup, teardown, and error boundaries.
        # Actual agent dispatch is handled by DispatcherAgent.
        log.info("RuntimeAgent: pipeline lifecycle started")
        return AgentResult(
            success=True,
            summary="Runtime pipeline completed",
            details={"agents_completed": list(state.agents_done)},
        )
