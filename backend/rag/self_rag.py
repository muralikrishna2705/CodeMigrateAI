"""Self-RAG — retrieve on demand, then reflect on what came back.

Two ideas from the Self-RAG paper, adapted to migration grounding:

1. **Retrieve-on-demand.** Not every query needs the corpus — a rename or a
   trivial syntax shift the model already knows is only diluted by injected
   reference text. Self-RAG first asks the model whether retrieval would help; if
   not, it returns no context and lets the model generate from its own knowledge.

2. **Reflection.** When it does retrieve, it grades each passage for relevance
   (the paper's ISREL token) and keeps only the passages that pass, so a single
   off-topic hit can't drag the whole context off course.

Both are conservative by construction for a grounding-critical tool: the
retrieve-on-demand gate is opt-in (``rag_self_rag_enabled``) and otherwise always
retrieves; reflection drops a passage only on an explicit "not relevant" verdict,
keeping it on any uncertainty or failure; and if reflection would empty the set,
the original hits are returned. No LLM → always retrieve, keep everything (i.e.
single-hop).
"""

import asyncio
import logging

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy

log = logging.getLogger("CodeMigrateAI.RAG.SelfRAG")

_RELEVANCE_SNIPPET_CHARS = 500


class SelfRAGStrategy(RetrievalStrategy):
    name = "self_rag"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        if not await self._should_retrieve(request):
            log.info("Self-RAG: model chose to generate without retrieval")
            return []

        hits = await self._run_query(request.query, request)
        if not hits or not self.has_llm:
            return hits

        # Reflect on passages in parallel; keep only those not explicitly rejected.
        verdicts = await asyncio.gather(
            *(self._reflect(request.query, doc, score) for doc, score in hits)
        )
        kept = [pair for pair in verdicts if pair is not None]
        # If reflection rejected everything, fall back to the raw hits rather than
        # hand the migrator no grounding at all — a weak grader over-rejects.
        return kept or hits

    async def _should_retrieve(self, request: RetrievalRequest) -> bool:
        """Decide whether retrieval is worth it. Defaults to True (safe)."""
        settings = get_settings()
        # The adaptive skip is opt-in; otherwise always ground (retrieval is the
        # anti-hallucination lever, so skipping it is the riskier default).
        if not getattr(settings, "rag_self_rag_enabled", False) or not self.has_llm:
            return True
        prompt = (
            "Would retrieving reference documentation help answer this migration "
            "question, or can it be answered from general knowledge of the "
            'languages? Respond as JSON: {"retrieve": true | false}.\n\n'
            f"QUESTION: {request.query}"
        )
        data = await self._ask_json(prompt, system="Output only the JSON verdict.")
        # Only skip on an explicit false; anything else (missing key, parse
        # failure) retrieves.
        return data.get("retrieve") is not False

    async def _reflect(self, query: str, doc, score):
        """Grade one passage; ``None`` to drop, ``(doc, score)`` to keep."""
        prompt = (
            f"Is the passage below relevant to answering this query?\n\n"
            f"QUERY: {query}\n\n"
            f"PASSAGE:\n{doc.page_content[:_RELEVANCE_SNIPPET_CHARS]}\n\n"
            'Respond as JSON: {"relevant": true | false}.'
        )
        data = await self._ask_json(prompt, system="Output only the JSON verdict.")
        if data.get("relevant") is False:
            return None
        return (doc, score)
