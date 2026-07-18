"""HyDE — Hypothetical Document Embeddings.

The retrieval query is normally the *question* ("what replaces ExecutorService
in Java 21?"). But the corpus stores *answers* — idiomatic target-language code
and docs — and a question embeds far from its answer. HyDE closes that gap: the
LLM first writes a hypothetical answer (a short target-language snippet that
*would* resolve the query), and we retrieve against that. An answer-shaped query
lands much nearer the answer-shaped documents in embedding space.

The hypothetical is prepended to — not substituted for — the original query, so
if the model produces something off, the real query terms (and the hybrid keyword
leg, which still runs on ``request.symbols``) keep retrieval grounded. With no
LLM wired, or on any failure, this degrades to a plain single-hop pass.
"""

import logging

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy

log = logging.getLogger("CodeMigrateAI.RAG.HyDE")

# Cap the hypothetical so it complements rather than drowns the real query in the
# embedded text — the same reasoning behind rag_query_code_chars for excerpts.
_MAX_HYPOTHETICAL_CHARS = 800


class HyDEStrategy(RetrievalStrategy):
    name = "hyde"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        if not getattr(settings, "rag_hyde_enabled", True) or not self.has_llm:
            return await self._run_query(request.query, request)

        hypothetical = await self._hypothesize(request)
        if not hypothetical:
            return await self._run_query(request.query, request)

        # Original query first (anchors the real intent), hypothetical answer
        # second (pulls the embedding toward answer-shaped documents).
        hyde_query = f"{request.query}\n\n{hypothetical}"
        return await self._run_query(hyde_query, request)

    async def _hypothesize(self, request: RetrievalRequest) -> str:
        """Ask the model for a short target-language snippet answering the query."""
        version = f" {request.target_version}" if request.target_version else ""
        prompt = (
            f"Write a short, idiomatic {request.target_language}{version} code "
            f"snippet that demonstrates the answer to this migration question:\n\n"
            f"{request.query}\n\n"
            "Output only code — no explanation, no prose."
        )
        raw = await self._ask(
            prompt,
            system=(
                f"You write concise {request.target_language} code. "
                "Output only code."
            ),
        )
        return raw.strip()[:_MAX_HYPOTHETICAL_CHARS]
