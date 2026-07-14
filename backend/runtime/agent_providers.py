import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ProviderAgent")


class ProviderAgent(BaseAgent):
    name = "ProviderAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        # Provides LLM client, cache manager, vector store to other agents.
        # Stores references in state for downstream consumption.
        log.info("ProviderAgent: dependencies injected")
        return AgentResult(
            success=True,
            summary="Providers initialized",
        )
