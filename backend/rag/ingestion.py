import asyncio
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

        from rag.url_index import (
            OFFICIAL_DOC_URLS,
            VERSIONED_DOC_TYPE,
            VERSIONED_DOC_URLS,
        )

        for lang in languages:
            log.info("Ingesting %s...", lang)

            # Step 1: Fetch web docs (Source 3)
            if settings.enable_web_docs and lang in OFFICIAL_DOC_URLS:
                await self.web_fetcher.fetch_for_language(lang, OFFICIAL_DOC_URLS[lang])

            # Step 1b: Fetch per-version official docs (What's New / migration
            # guides) into _fetched/<version>/<doc_type>/. This is what supplies
            # the version-aware retrieval ladder with real versioned content.
            if settings.enable_web_docs and lang in VERSIONED_DOC_URLS:
                await self.web_fetcher.fetch_versioned(
                    lang, VERSIONED_DOC_URLS[lang], VERSIONED_DOC_TYPE
                )

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
                    # Chroma writes, the embedding call behind them, and the
                    # rate-limiter/backoff sleeps inside that call are all
                    # blocking. Ingestion runs as a background task on the
                    # server's event loop, so doing this inline froze the whole
                    # worker — /health and /migrate included — for the length of
                    # every quota backoff. Same treatment as every other store
                    # access (see runtime/agent_observer.py).
                    await asyncio.to_thread(
                        self.vector_store.add_documents, unique_chunks
                    )
                    log.info("  Ingested %d unique chunks for %s", len(unique_chunks), lang)
                else:
                    log.info("  No new chunks for %s", lang)

    async def close(self):
        await self.web_fetcher.close()
