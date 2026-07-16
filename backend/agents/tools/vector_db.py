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
from config import get_settings


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
    }

    def __init__(self, rag_pipeline, timeout_sec: float | None = None) -> None:
        super().__init__(timeout_sec)
        self._rag = rag_pipeline

    async def run(
        self,
        query: str = "",
        target_language: str = "",
        target_version: str = "",
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

        hits = await self._rag.search(
            query=query,
            target_language=target_language,
            target_version=target_version,
        )
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

    @staticmethod
    def format_context(hits: list[tuple]) -> str:
        """Render hits in the same shape ``enrich_prompt`` produces.

        The MigratorAgent keys off the literal "Reference Examples" heading (and
        RetrieverAgent checks for it before accepting context), so tool-retrieved
        context must render identically to passively-retrieved context or it
        would be silently dropped downstream.
        """
        settings = get_settings()
        parts = [
            "## Reference Examples\nHere are relevant code patterns from the "
            "target language:\n"
        ]
        for doc, score in hits:
            md = doc.metadata or {}
            lang = md.get("language", "unknown")
            version = md.get("version", "")
            doc_type = md.get("doc_type", "")
            label_bits = [lang]
            if version and version != settings.rag_version_wildcard:
                label_bits.append(version)
            if doc_type:
                label_bits.append(doc_type)
            parts.append(f"### {' · '.join(label_bits)} (relevance: {score:.2f})")
            parts.append(f"```{lang}\n{doc.page_content}\n```")
        return "\n\n".join(parts)
