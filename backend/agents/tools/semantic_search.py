"""SemanticSearchTool — search past migrations across sessions.

Answers "have we migrated code like this before, and what did we do?" against
:class:`~rag.migration_memory.SemanticMigrationMemory`, which persists validated
migrations to disk and therefore outlives the process.

Distinct from ``vector_db``: that searches documentation (how the language works),
this searches experience (what this system already did successfully). A hit here
is a strong planning signal — a known-good precedent for the same construct — but
it is precedent, not authority, so it never displaces official docs.

Off by default (``tool_semantic_search_enabled``): the memory collection is empty
until migrations have run, and searching an empty store just burns an embedding
call per query.
"""

from agents.tools.base import AgentTool, ToolResult
from config import get_settings


class SemanticSearchTool(AgentTool):
    name = "semantic_search"
    description = (
        "Search previously completed migrations for precedents similar to the "
        "current code. Use to reuse an approach that already validated cleanly."
    )
    parameters = {
        "query": "code or description of the construct to find precedents for",
        "target_language": "restrict to migrations into this language (optional)",
        "target_version": "restrict to migrations into this version (optional)",
    }

    def __init__(self, memory, timeout_sec: float | None = None) -> None:
        super().__init__(timeout_sec)
        self._memory = memory

    async def run(
        self,
        query: str = "",
        target_language: str = "",
        target_version: str = "",
        **_,
    ) -> ToolResult:
        if not self._memory:
            return ToolResult(
                tool=self.name, success=False, error="Migration memory not available"
            )
        if not query.strip():
            return ToolResult(tool=self.name, success=False, error="query is required")

        where = self._build_filter(target_language, target_version)

        # Chroma calls are blocking; keep them off the event loop the same way
        # RAGPipeline does for the reference store.
        import asyncio

        settings = get_settings()
        hits = await asyncio.to_thread(
            self._memory.search, query, settings.memory_top_k, where
        )
        hits = [
            (doc, score) for doc, score in hits if score >= settings.memory_min_score
        ]

        if not hits:
            return ToolResult(
                tool=self.name,
                success=True,
                data={"query": query, "precedents": []},
                summary="No similar past migrations found",
            )

        precedents = [
            {
                "score": score,
                "source_language": (doc.metadata or {}).get("source_language", ""),
                "target_language": (doc.metadata or {}).get("language", ""),
                "target_version": (doc.metadata or {}).get("version", ""),
                "plan_summary": (doc.metadata or {}).get("plan_summary", ""),
                "migrated_code": (doc.metadata or {}).get("migrated_code", ""),
            }
            for doc, score in hits
        ]
        return ToolResult(
            tool=self.name,
            success=True,
            data={"query": query, "precedents": precedents},
            summary=f"{len(precedents)} similar past migration(s)",
        )

    @staticmethod
    def _build_filter(target_language: str, target_version: str) -> dict | None:
        """Chroma metadata filter. Chroma requires ``$and`` for multiple keys."""
        clauses = []
        if target_language:
            clauses.append({"language": target_language})
        if target_version:
            clauses.append({"version": target_version})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
