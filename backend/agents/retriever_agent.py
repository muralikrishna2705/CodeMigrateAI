import logging

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RetrieverAgent")


class RetrieverAgent(BaseAgent):
    name = "RetrieverAgent"
    requires_llm = False
    needs = ("rag_pipeline",)

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self._rag_pipeline = config.get("rag_pipeline") if config else None

    async def run(self, state: MigrationState) -> AgentResult:
        if not self._rag_pipeline:
            return AgentResult(
                success=True, summary="RAG pipeline not available, skipping"
            )

        settings = get_settings()
        # Feedback bus: targeted queries enqueued by an upstream agent (e.g. the
        # MigratorAgent's ungrounded imports). Optionally add LLM-expanded terms.
        extra_terms = list(state.retrieval_requests)
        if settings.rag_query_expansion:
            extra_terms += await self._expand_query_terms(state)

        enriched = await self._rag_pipeline.enrich_prompt(
            source_language=state.source_language,
            target_language=state.target_language,
            source_code=state.source_code,
            base_prompt="",  # Consumed by MigratorAgent via state.rag_context
            target_version=state.target_version,
            code_metrics=state.code_metrics,
            extra_terms=extra_terms,
        )
        state.rag_context = enriched if "Reference Examples" in enriched else ""

        summary = (
            "Retrieved RAG context" if state.rag_context else "No relevant context found"
        )
        if state.retrieval_requests:
            summary += f" · re-retrieval for {len(state.retrieval_requests)} import(s)"
        return AgentResult(
            success=True,
            summary=summary,
            details={
                "context_length": len(state.rag_context),
                "reretrieval_terms": list(state.retrieval_requests),
            },
        )

    async def _expand_query_terms(self, state: MigrationState) -> list[str]:
        """Reformulate code signals into targeted query terms via the fast model.

        Best-effort: any failure (stub LLM, model down, bad output) returns an
        empty list so retrieval falls back to the concatenated-signal query.
        """
        call_llm = getattr(self.llm, "call_llm", None)
        if call_llm is None:
            return []
        try:
            prompt = (
                f"A developer is migrating {state.source_language} "
                f"{state.source_version} code to {state.target_language} "
                f"{state.target_version}. Based on the code below, list up to 6 "
                "short search keywords (comma-separated, no sentences) naming the "
                "libraries, APIs, and language constructs whose migration matters "
                "most.\n\nCODE:\n"
                f"{state.source_code[: get_settings().max_llm_code_chars]}"
            )
            raw = await call_llm(
                prompt,
                system_prompt="Output only a comma-separated keyword list.",
                **self._fast_model_kwargs(),
            )
            terms = [t.strip() for t in raw.replace("\n", ",").split(",")]
            return [t for t in terms if t and len(t) <= 40][:6]
        except Exception as exc:  # noqa: BLE001 — expansion is strictly optional
            log.warning("Query expansion failed: %s", exc)
            return []
