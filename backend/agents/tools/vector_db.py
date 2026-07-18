"""VectorDBTool — on-demand RAG query with an agent-formulated question.

The distinction from the existing retrieval path matters. ``RetrieverAgent``
historically ran once, built a query from code-signal heuristics (imports, call
names, type names), and injected whatever came back into ``state.rag_context``
whether or not it was relevant — passive injection. This tool inverts that: an
agent asks a *specific question* ("what replaces ExecutorService in Java 21?")
and gets ranked hits back, so it can ask again with a better query if the first
answer is thin.

Both paths share ``RAGPipeline``'s filter ladder, ranking, and cache, so
on-demand queries are grounded exactly the way the passive ones are.
"""

from agents.tools.base import AgentTool, ToolResult
from rag.retrieval_pipeline import RAGPipeline, RetrievalRequest


class VectorDBTool(AgentTool):
    name = "vector_db"
    description = (
        "Search the reference corpus of language documentation and migration "
        "guides. Use to find how a specific API, library, or construct is "
        "written in the target language."
    )
    parameters = {
        "query": "what to search for, as a specific question or API/symbol name",
        "target_language": "language to restrict results to (optional)",
        "target_version": "language version to prefer (optional)",
        "intent": (
            "retrieval style (optional): 'precise' for an exact answer (HyDE), "
            "'exploratory' to cast wide (multi-query), 'verify' for a plain lookup"
        ),
    }

    # Maps the agent's stated intent to the retrieval strategy that serves it:
    # a precise question benefits from an answer-shaped HyDE query; an open-ended
    # exploration from multi-query fan-out; a simple existence check from a plain
    # single pass. Unknown/empty intent → the default single-hop search.
    _INTENT_STRATEGY = {
        "precise": "hyde",
        "exploratory": "multi_query",
        "verify": "single_hop",
    }

    def __init__(self, rag_pipeline, timeout_sec: float | None = None) -> None:
        super().__init__(timeout_sec)
        self._rag = rag_pipeline

    async def run(
        self,
        query: str = "",
        target_language: str = "",
        target_version: str = "",
        intent: str = "",
        **_,
    ) -> ToolResult:
        if not self._rag:
            return ToolResult(
                tool=self.name,
                success=False,
                error="RAG pipeline not available",
            )
        if not query.strip():
            return ToolResult(
                tool=self.name, success=False, error="query is required"
            )

        hits = await self._retrieve(query, target_language, target_version, intent)
        if not hits:
            return ToolResult(
                tool=self.name,
                success=True,
                data={"query": query, "hits": [], "context": ""},
                summary=f"No corpus matches for {query!r}",
            )

        return ToolResult(
            tool=self.name,
            success=True,
            data={
                "query": query,
                "hits": [
                    {
                        "content": doc.page_content,
                        "score": score,
                        "language": (doc.metadata or {}).get("language", "unknown"),
                        "version": (doc.metadata or {}).get("version", ""),
                        "doc_type": (doc.metadata or {}).get("doc_type", ""),
                        "is_official": bool((doc.metadata or {}).get("is_official")),
                    }
                    for doc, score in hits
                ],
                "context": self.format_context(hits),
            },
            summary=f"{len(hits)} reference match(es) for {query!r}",
        )

    async def _retrieve(
        self, query: str, target_language: str, target_version: str, intent: str
    ) -> list[tuple]:
        """Route by intent to an agentic strategy, or the default search.

        An intent maps to a strategy (see ``_INTENT_STRATEGY``) and goes through
        the pipeline's ``retrieve``; without an intent — or against a pipeline
        that predates strategy support — it falls back to the plain ``search``,
        which is the historical behaviour. Either way results share the same
        filter ladder, ranking, and cache.
        """
        strategy = self._INTENT_STRATEGY.get(intent.strip().lower()) if intent else None
        if strategy and hasattr(self._rag, "retrieve"):
            request = RetrievalRequest(
                query=query,
                target_language=target_language,
                target_version=target_version,
            )
            return await self._rag.retrieve(request, strategy=strategy)
        return await self._rag.search(
            query=query,
            target_language=target_language,
            target_version=target_version,
        )

    @staticmethod
    def format_context(hits: list[tuple]) -> str:
        """Render hits in the same shape ``enrich_prompt`` produces.

        The MigratorAgent keys off the literal "Reference Examples" heading (and
        RetrieverAgent checks for it before accepting context), so tool-retrieved
        context must render identically to passively-retrieved context or it
        would be silently dropped downstream. Both now delegate to the single
        renderer on RAGPipeline so the shape can never drift between paths.
        """
        return RAGPipeline.render_reference_context(hits)
