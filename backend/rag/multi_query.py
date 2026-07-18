"""Multi-Query retrieval.

A single query phrasing only ever probes one region of the embedding space; a
document phrased differently from the query is missed even when it's exactly on
topic. Multi-Query has the LLM rewrite the query into several paraphrases
(synonyms, framework-specific terms, different granularities), retrieves for each
in parallel, and fuses the results — so recall no longer hinges on the caller
having guessed the corpus's wording.

Fusion dedups by content (``merge_hits``), so a document surfacing for several
variants is kept once at its best score. With no LLM, or if variant generation
fails, this collapses to a single-hop pass on the original query.
"""

import asyncio
import logging

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy, merge_hits

log = logging.getLogger("CodeMigrateAI.RAG.MultiQuery")


class MultiQueryStrategy(RetrievalStrategy):
    name = "multi_query"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        count = getattr(settings, "rag_multi_query_count", 5)

        variants = await self._variants(request, count) if self.has_llm else []
        # Always include the original query; dedup case-insensitively so a variant
        # that merely restates it doesn't cost an extra retrieval.
        queries: list[str] = []
        seen: set[str] = set()
        for q in [request.query, *variants]:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                queries.append(q)

        hitlists = await asyncio.gather(
            *(self._run_query(q, request) for q in queries)
        )
        return merge_hits(list(hitlists), settings.rag_top_k)

    async def _variants(self, request: RetrievalRequest, count: int) -> list[str]:
        """Ask the model for ``count`` alternative phrasings, one per line."""
        version = f" {request.target_version}" if request.target_version else ""
        prompt = (
            f"A developer is migrating to {request.target_language}{version}. "
            f"Rewrite the search query below into {count} alternative phrasings "
            "that would retrieve relevant documentation — vary the wording, use "
            "synonyms and specific API/framework terms. One query per line, no "
            "numbering.\n\n"
            f"QUERY: {request.query}"
        )
        raw = await self._ask(
            prompt, system="Output one search query per line. No numbering, no prose."
        )
        variants: list[str] = []
        for line in raw.splitlines():
            # Strip list markers/numbering the model adds despite instructions.
            cleaned = line.strip().lstrip("-*0123456789.) \t").strip()
            if cleaned:
                variants.append(cleaned)
        return variants[:count]
