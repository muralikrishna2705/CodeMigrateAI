import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

log = logging.getLogger("CodeMigrateAI.VectorStore")

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "codemigrate_ref"


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

    def add_documents(self, documents: list[Document]):
        if not self._store:
            self.initialize()
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        # Batch to avoid memory issues
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            self._store.add_texts(
                texts[i:i + batch_size],
                metadatas[i:i + batch_size] if metadatas else None,
            )
        log.info("Added %d documents to vector store", len(documents))

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
