"""Cross-encoder reranking — the precision stage between retrieval and the prompt.

Everything upstream ranks documents *without ever comparing them to the query
directly*. The dense leg scores a query embedding against document embeddings,
which is a lossy proxy. Reciprocal rank fusion is worse in this respect: it
combines *positions*, so a document ranked #1 by a keyword leg that matched one
incidental symbol fuses as strongly as a genuinely relevant one.

A cross-encoder reads the query and the document together and scores that pair.
It is the difference between "these embeddings are near each other" and "this
document answers this question", and it is the single largest precision lever
available short of improving the corpus itself.

Because it is accurate, it changes the retrieval shape upstream: fetch *wide*
(``rag_rerank_candidates``) and let the reranker cut to ``rag_top_k``, rather
than fetching narrowly and hoping the cosine floor was set correctly. That is
also why ``rag_min_score`` can now be low — the floor is no longer what protects
the prompt from junk.

FlashRank runs locally on CPU (~150-200ms for 20 documents) and costs no API
call, so this stays affordable on the hot path.
"""

import asyncio
import logging
import threading

from config import get_settings

log = logging.getLogger("CodeMigrateAI.Reranker")

_lock = threading.Lock()
_reranker = None
_unavailable = False


def get_reranker():
    """The shared reranker, or None when unavailable/disabled.

    Loading pulls a model from disk (and downloads it once, on first use), so it
    is built once per process and cached. A failure is cached too — retrying a
    missing model on every query would add its failure latency to every retrieval.
    """
    global _reranker, _unavailable
    settings = get_settings()
    if not settings.rag_rerank_enabled or _unavailable:
        return None
    if _reranker is not None:
        return _reranker

    with _lock:
        if _reranker is None and not _unavailable:
            try:
                from langchain_community.document_compressors import FlashrankRerank

                _reranker = FlashrankRerank(
                    model=settings.rag_rerank_model, top_n=settings.rag_top_k
                )
                log.info("Reranker ready: %s", settings.rag_rerank_model)
            except Exception as exc:  # noqa: BLE001 — reranking is an enhancement
                _unavailable = True
                log.warning(
                    "Reranker unavailable (%s); falling back to fusion order", exc
                )
    return _reranker


async def rerank(hits: list[tuple], query: str, top_n: int | None = None) -> list[tuple]:
    """Re-score ``(doc, score)`` hits against ``query``, best first.

    Returns the input unchanged when reranking is off, unavailable, or there is
    nothing worth reordering — so callers can apply this unconditionally and
    retrieval never gets *worse* than the fusion order it already had.

    The returned scores are the cross-encoder's, not the original cosine/RRF
    values: downstream ranking adds metadata boosts on top, and mixing two
    incomparable scales there would make those boosts mean different things for
    different documents.
    """
    if len(hits) < 2:
        return hits
    reranker = get_reranker()
    if reranker is None or not query.strip():
        return hits

    settings = get_settings()
    limit = top_n or settings.rag_top_k
    docs = [doc for doc, _ in hits]

    try:
        # FlashRank is synchronous and CPU-bound; off-thread so it cannot stall
        # the event loop while other branches of the graph are in flight.
        reranked = await asyncio.to_thread(
            _compress, reranker, docs, query, limit
        )
    except Exception as exc:  # noqa: BLE001 — degrade to the incoming order
        log.warning("Reranking failed, keeping fusion order: %s", exc)
        return hits

    if not reranked:
        return hits

    out = [
        (doc, float((doc.metadata or {}).get("relevance_score", 0.0)))
        for doc in reranked
    ]
    log.info(
        "Reranked %d -> %d (top score %.3f)", len(hits), len(out), out[0][1] if out else 0.0
    )
    return out


def _compress(reranker, docs, query: str, limit: int):
    """Run the reranker with ``top_n`` scoped to this call.

    ``FlashrankRerank.top_n`` is set at construction, but the useful cut differs
    per call site (a tool query wants fewer than a full enrich pass). Setting it
    around the call keeps one cached model serving both.
    """
    previous = reranker.top_n
    reranker.top_n = limit
    try:
        return reranker.compress_documents(docs, query)
    finally:
        reranker.top_n = previous


def reset() -> None:
    """Drop the cached reranker (tests changing rerank settings)."""
    global _reranker, _unavailable
    with _lock:
        _reranker = None
        _unavailable = False
