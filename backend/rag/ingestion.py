import hashlib
import logging

from config import get_settings

from .document_pipeline import DocumentPipeline
from .loaders import DocLoader
from .splitter import DocSplitter
from .web_doc_fetcher import WebDocFetcher

log = logging.getLogger("CodeMigrateAI.Ingestion")

SOURCE_TYPES = ["builtin", "user", "fetched"]


class IngestionPipeline:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.doc_pipeline = DocumentPipeline(DocLoader(), DocSplitter())
        self.web_fetcher = WebDocFetcher()
        self._ingested_hashes: set[str] = set()

    async def run(self, languages: list[str]):
        settings = get_settings()
        if not settings.enable_rag:
            log.info("RAG disabled, skipping ingestion")
            return

        from rag.url_index import OFFICIAL_DOC_URLS

        for lang in languages:
            log.info("Ingesting %s...", lang)

            # Step 1: Fetch web docs (Source 3)
            if settings.enable_web_docs and lang in OFFICIAL_DOC_URLS:
                await self.web_fetcher.fetch_for_language(lang, OFFICIAL_DOC_URLS[lang])

            # Step 2: Load, split, embed all 3 sources
            chunks = await self.doc_pipeline.process_language(lang, SOURCE_TYPES)
            if chunks:
                # Deduplicate by content hash
                unique_chunks = []
                for chunk in chunks:
                    h = hashlib.sha256(chunk.page_content.encode()).hexdigest()
                    if h not in self._ingested_hashes:
                        self._ingested_hashes.add(h)
                        unique_chunks.append(chunk)
                if unique_chunks:
                    self.vector_store.add_documents(unique_chunks)
                    log.info("  Ingested %d unique chunks for %s", len(unique_chunks), lang)
                else:
                    log.info("  No new chunks for %s", lang)

    async def close(self):
        await self.web_fetcher.close()
