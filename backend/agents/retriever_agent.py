import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RetrieverAgent")


class RetrieverAgent(BaseAgent):
    name = "RetrieverAgent"
    requires_llm = False

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self._rag_pipeline = config.get("rag_pipeline") if config else None

    async def run(self, state: MigrationState) -> AgentResult:
        if not self._rag_pipeline:
            return AgentResult(
                success=True, summary="RAG pipeline not available, skipping"
            )

        enriched = await self._rag_pipeline.enrich_prompt(
            source_language=state.source_language,
            target_language=state.target_language,
            source_code=state.source_code,
            base_prompt="",  # Consumed by MigratorAgent via state.rag_context
        )
        state.rag_context = enriched if "Reference Examples" in enriched else ""
        return AgentResult(
            success=True,
            summary="Retrieved RAG context" if state.rag_context else "No relevant context found",
            details={"context_length": len(state.rag_context)},
        )
