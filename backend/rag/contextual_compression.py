"""Contextual compression of retrieved documents.

Retrieved chunks are sized for recall (2000 chars), but most of a chunk is
usually noise for any given query — and every noise token spent on reference
context is a token not spent on the code being migrated, and a distraction the
model may latch onto. This strategy runs an LLM extractor over each hit (the
native equivalent of LangChain's ``LLMChainExtractor``), keeping only the
sentences/lines relevant to the query. Documents that compress to nothing are
dropped entirely, which doubles as a lightweight relevance filter.

Opt-in (``rag_compression_enabled``) because it costs one LLM call per hit. It is
also strictly non-destructive on failure: a document that errors or comes back
empty is kept at full length rather than lost, so compression can only ever
tighten context, never drop grounding by accident.
"""

import asyncio
import logging

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy

log = logging.getLogger("CodeMigrateAI.RAG.Compression")

# The model emits this sentinel for a document with nothing on-topic; such a
# document is dropped (its retrieval was a false positive).
_DROP_SENTINEL = "NONE"


class ContextualCompressionStrategy(RetrievalStrategy):
    name = "contextual_compression"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        hits = await self._run_query(request.query, request)
        # Compression is opt-in and needs a model; without either, return the raw
        # hits unchanged so behavior matches single-hop.
        if not hits or not self.has_llm or not getattr(
            settings, "rag_compression_enabled", False
        ):
            return hits

        compressed = await asyncio.gather(
            *(self._compress(doc, score, request) for doc, score in hits)
        )
        return [pair for pair in compressed if pair is not None]

    async def _compress(self, doc, score, request: RetrievalRequest):
        """Extract query-relevant lines from one doc.

        Returns ``(doc, score)`` with trimmed content, ``None`` to drop an
        irrelevant doc, or the original pair unchanged on any failure.
        """
        content = doc.page_content
        prompt = (
            "From the reference document below, extract ONLY the lines directly "
            f"relevant to answering this query:\n\n{request.query}\n\n"
            f"Return the extracted lines verbatim. If nothing in the document is "
            f"relevant, return exactly {_DROP_SENTINEL}.\n\n"
            f"DOCUMENT:\n{content}"
        )
        extracted = (
            await self._ask(
                prompt,
                system="Extract relevant lines verbatim, or output NONE.",
            )
        ).strip()

        if not extracted:
            # LLM unavailable / empty response → keep the doc intact (don't drop
            # grounding because the extractor was silent).
            return (doc, score)
        if extracted.upper().startswith(_DROP_SENTINEL):
            return None

        # Mutate in place — the doc is a fresh object from this retrieval, and this
        # avoids depending on the concrete Document class (works for test fakes too).
        doc.page_content = extracted
        return (doc, score)
