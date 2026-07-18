"""Parent Document retrieval.

There's a tension in chunking: small chunks embed precisely (a tight chunk about
one API matches a query about that API cleanly), but small chunks are missing the
surrounding context a model needs to actually use them. Parent Document retrieval
resolves it — retrieve on the small, precise child chunks, then return the *whole
parent document* they came from, so the model gets both the precise match and its
full context.

Children are linked to parents at ingest time via ``parent_id`` + ``chunk_index``
metadata (see ``rag.splitter``). Sibling chunks sharing a ``parent_id`` are pulled
back and reassembled in ``chunk_index`` order. Chunks ingested before that metadata
existed have no ``parent_id`` and are returned as-is — so this degrades cleanly to
child-chunk retrieval on an older corpus.
"""

import asyncio
import logging

from config import get_settings
from rag.retrieval_pipeline import RetrievalRequest, RetrievalStrategy

log = logging.getLogger("CodeMigrateAI.RAG.ParentDoc")


class ParentDocumentStrategy(RetrievalStrategy):
    name = "parent_document"

    async def retrieve(self, request: RetrievalRequest) -> list[tuple]:
        settings = get_settings()
        hits = await self._run_query(request.query, request)
        if not hits:
            return hits

        get_by_metadata = getattr(self.pipeline.vector_store, "get_by_metadata", None)
        if get_by_metadata is None:
            # Store can't look up siblings → return the child chunks unchanged.
            return hits

        out: list[tuple] = []
        seen_parents: set[str] = set()
        for doc, score in hits:
            parent_id = (doc.metadata or {}).get("parent_id")
            if not parent_id:
                # Legacy chunk without parent linkage — keep as-is.
                out.append((doc, score))
                continue
            if parent_id in seen_parents:
                # A sibling of this parent already surfaced; the parent it expands
                # to already contains this chunk, so skip the duplicate.
                continue
            seen_parents.add(parent_id)

            try:
                siblings = await asyncio.to_thread(
                    get_by_metadata, {"parent_id": parent_id}
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Parent lookup failed for %s: %s", parent_id, exc)
                siblings = []

            if len(siblings) <= 1:
                out.append((doc, score))
                continue

            siblings.sort(key=lambda d: (d.metadata or {}).get("chunk_index", 0))
            # Keep the child's score (its precise match is the relevance signal)
            # and metadata, but widen its content to the full parent document.
            doc.page_content = "\n".join(s.page_content for s in siblings)
            out.append((doc, score))

        return out[: settings.rag_top_k]
