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
            embedding_function=self._embedding_service._inner,
            persist_directory=str(PERSIST_DIR),
            collection_metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 100,
                "hnsw:M": 16,
                "hnsw:search_ef": 50,
            },
        )
        count = self._store._collection.count()
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

    def similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.7):
        if not self._store:
            self.initialize()
        docs_with_scores = self._store.similarity_search_with_relevance_scores(query, k=k)
        # Filter by threshold
        filtered = [(doc, score) for doc, score in docs_with_scores if score >= score_threshold]
        if not filtered and docs_with_scores:
            log.warning("All %d results below threshold %.2f, returning empty", len(docs_with_scores), score_threshold)
        return filtered

    def count(self) -> int:
        if not self._store:
            return 0
        return self._store._collection.count()
