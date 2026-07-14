import logging
import time

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RuntimeAgent")


class RuntimeAgent(BaseAgent):
    name = "RuntimeAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        start = time.perf_counter()
        agent_count = len(state.agents_done)
        error_count = len(state.errors)
        duration = time.perf_counter() - start

        log.info(
            "RuntimeAgent: %d agents completed, %d errors in %.2fs",
            agent_count,
            error_count,
            duration,
        )
        return AgentResult(
            success=True,
            summary=f"Pipeline: {agent_count} agents, {error_count} errors",
            details={
                "agents_completed": list(state.agents_done),
                "error_count": error_count,
                "duration_seconds": round(duration, 3),
            },
        )
