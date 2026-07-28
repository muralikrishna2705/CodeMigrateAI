import hashlib
import logging
from pathlib import Path

from config import get_settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

log = logging.getLogger("CodeMigrateAI.VectorStore")

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "codemigrate_ref"


def content_id(text: str) -> str:
    """A stable Chroma id derived from the chunk's content.

    Ingestion used to let Chroma mint a fresh UUID per insert, so every restart
    appended a complete second copy of the corpus — the store grew without bound
    and re-embedded (re-paid for) content it already held. A content hash makes
    the write idempotent: ``add_texts`` upserts, so re-ingesting the same chunk
    overwrites its own row instead of creating a new one, and an unchanged
    corpus can be skipped entirely.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class VectorStore:
    def __init__(self, embedding_service):
        self._embedding_service = embedding_service
        self._store: Chroma | None = None

    def initialize(self):
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embedding_service,
            persist_directory=str(PERSIST_DIR),
            collection_metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:M": 16,
                "hnsw:search_ef": 50,
            },
        )
        count = len(self._store.get()["ids"]) if self._store else 0
        log.info("Vector store initialized with %d existing docs", count)

    def _existing_ids(self, ids: list[str]) -> set[str]:
        """Which of ``ids`` the collection already holds.

        Best-effort: a store that cannot answer is treated as holding nothing,
        which costs a re-embed but never skips a document that is actually
        missing.
        """
        if not ids:
            return set()
        try:
            return set(self._store.get(ids=ids, include=[])["ids"])
        except Exception as exc:  # noqa: BLE001 — an unanswerable store re-embeds
            log.debug("Existing-id lookup failed, assuming none: %s", exc)
            return set()

    def add_documents(self, documents: list[Document]):
        """Index ``documents``, skipping what is already stored or unembeddable.

        Two things this deliberately does not do, both of which it used to:

        * **Persist a failed embedding.** The embedding service must return one
          vector per text to satisfy the interface, so it pads failures with a
          zero vector. Zero vectors are the right answer for a transient query
          (they match nothing) and a silent disaster in a store: permanently
          unretrievable, yet counted as indexed and reported as success. Each
          batch is therefore embedded here first, and only texts that came back
          with a real vector are handed to Chroma.
        * **Re-pay for content it already has.** Ids are content hashes, so the
          already-stored chunks are filtered out before any embedding call.

        The pre-flight embed is not wasted work: the embedding service caches
        successes, so the ``add_texts`` call behind it is served from that cache
        rather than making a second API call. Batches are capped to the cache
        capacity so that stays true.
        """
        if not self._store:
            self.initialize()
        if not documents:
            return

        settings = get_settings()
        batch_size = max(1, settings.embed_batch_size)
        capacity = getattr(self._embedding_service, "cache_capacity", batch_size)
        if capacity and batch_size > capacity:
            log.debug(
                "Capping write batch %d -> %d to fit the embedding cache",
                batch_size,
                capacity,
            )
            batch_size = capacity

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = [content_id(t) for t in texts]

        # Two chunks with identical content collapse to one id; Chroma rejects a
        # single upsert that repeats an id, so de-duplicate before batching.
        pending: list[int] = []
        seen: set[str] = set()
        for i, doc_id in enumerate(ids):
            if doc_id not in seen:
                seen.add(doc_id)
                pending.append(i)

        already = self._existing_ids([ids[i] for i in pending])
        if already:
            pending = [i for i in pending if ids[i] not in already]
            log.info("Skipping %d chunks already indexed", len(already))
        if not pending:
            log.info("Nothing new to index (%d chunks already present)", len(documents))
            return

        added = 0
        failed = 0
        for start in range(0, len(pending), batch_size):
            window = pending[start:start + batch_size]
            batch_texts = [texts[i] for i in window]

            self._embedding_service.embed_documents(batch_texts)
            unembeddable = getattr(
                self._embedding_service, "last_failed_texts", set()
            )
            good = [i for i in window if texts[i] not in unembeddable]
            failed += len(window) - len(good)
            if not good:
                continue

            self._store.add_texts(
                [texts[i] for i in good],
                [metadatas[i] for i in good] if metadatas else None,
                ids=[ids[i] for i in good],
            )
            added += len(good)

        if failed:
            # Loud on purpose: a partial index is a silent quality regression at
            # retrieval time, and the count is the only signal it happened.
            log.warning(
                "Indexed %d documents; %d could not be embedded and were NOT "
                "stored (they will be retried on the next ingestion)",
                added,
                failed,
            )
        else:
            log.info("Added %d documents to vector store", added)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.7,
        where: dict | None = None,
    ):
        if not self._store:
            self.initialize()
        # `where` is a Chroma metadata filter (e.g. {"language": "python"}) used
        # to restrict retrieval to a subset of docs; omitted -> search all docs.
        kwargs = {"k": k}
        if where:
            kwargs["filter"] = where
        try:
            docs_with_scores = self._store.similarity_search_with_relevance_scores(
                query, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 — retrieval degrades, never fails
            # Retrieval is grounding, not correctness: a migration without it is
            # worse, not broken, and the caller already handles no hits by
            # emitting the ungrounded notice. Raising here would turn a
            # transient embedding outage into a failed migration.
            #
            # The concrete case: the embedding service returns 503,
            # CachedEmbeddings substitutes a zero vector of its *assumed* width,
            # and Chroma rejects it outright when the collection was built at a
            # different width ("expecting dimension 3072, got 768"). The
            # fallback exists to degrade gracefully and without this it crashed.
            log.warning("Vector search failed, continuing without it: %s", exc)
            return []
        # Filter by threshold
        filtered = [(doc, score) for doc, score in docs_with_scores if score >= score_threshold]
        if not filtered and docs_with_scores:
            log.warning("All %d results below threshold %.2f, returning empty", len(docs_with_scores), score_threshold)
        return filtered

    def keyword_search(
        self, symbols: list[str], k: int = 4, where: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Exact-symbol keyword leg for hybrid retrieval (no embedding cost).

        Uses Chroma's ``where_document`` ``$contains`` to find docs that literally
        contain each query symbol, then ranks by how many distinct symbols each
        doc matches. Returns ``(doc, match_ratio)`` where match_ratio is
        matched/queried symbols — complements the dense vector leg for exact
        API/type names that embeddings tend to miss.
        """
        if not self._store:
            self.initialize()
        if not symbols:
            return []

        limit = max(k * 5, 20)
        matches: dict[str, list] = {}  # doc id -> [Document, match_count]
        for symbol in symbols:
            try:
                res = self._store.get(
                    where_document={"$contains": symbol},
                    where=where or None,
                    limit=limit,
                    include=["documents", "metadatas"],
                )
            except Exception as exc:
                log.debug("keyword get failed for %r: %s", symbol, exc)
                continue
            documents = res.get("documents") or []
            metadatas = res.get("metadatas") or []
            ids = res.get("ids") or []
            for i, content in enumerate(documents):
                doc_id = ids[i] if i < len(ids) else content
                if doc_id not in matches:
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    matches[doc_id] = [Document(page_content=content, metadata=metadata), 0]
                matches[doc_id][1] += 1

        total = len(symbols)
        ranked = sorted(matches.values(), key=lambda m: m[1], reverse=True)[:k]
        return [(doc, count / total) for doc, count in ranked]

    def get_by_metadata(
        self, where: dict, limit: int = 50
    ) -> list[Document]:
        """Fetch documents by a metadata filter (no embedding/query involved).

        Backs the Parent Document strategy: given a chunk's ``parent_id``, pull
        every sibling chunk so they can be reassembled into the full parent doc.
        Returns an empty list on any store error — the caller then degrades to the
        original child chunk.
        """
        if not self._store:
            self.initialize()
        if not where:
            return []
        try:
            res = self._store.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            log.debug("get_by_metadata failed for %s: %s", where, exc)
            return []
        documents = res.get("documents") or []
        metadatas = res.get("metadatas") or []
        return [
            Document(
                page_content=content,
                metadata=metadatas[i] if i < len(metadatas) else {},
            )
            for i, content in enumerate(documents)
        ]

    def count(self) -> int:
        if not self._store:
            return 0
        return len(self._store.get()["ids"])
