"""Build the Chroma reference index offline.

The app ingests in a background task on startup, which is right for a server
(the API answers immediately and RAG attaches when it is ready) and wrong for a
demo: the first migration of a fresh checkout retrieves nothing, because
embedding the corpus has not finished. Building the index ahead of time makes it
a deliverable artifact instead of a race.

Usage, from the repo root::

    python backend/scripts/build_index.py               # every language
    python backend/scripts/build_index.py python java   # a subset
    python backend/scripts/build_index.py --no-web      # skip doc fetching
    python backend/scripts/build_index.py --stats       # report and exit

Reads the same settings as the app, so the embedding provider and model come
from ``.env``. With ``EMBEDDING_PROVIDER=google_genai`` this needs a network
connection and an API key.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Run as a script from anywhere: the backend packages are top-level (see the
# pythonpath setting in pyproject.toml) rather than installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from llm import providers  # noqa: E402
from rag.embedding_service import CachedEmbeddings  # noqa: E402
from rag.ingestion import IngestionPipeline  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402

log = logging.getLogger("build_index")


def _build_store() -> tuple[VectorStore, CachedEmbeddings]:
    settings = get_settings()
    embeddings = CachedEmbeddings(
        providers.get_embeddings(settings),
        max_cache=settings.local_cache_max_entries,
    )
    store = VectorStore(embeddings)
    store.initialize()
    return store, embeddings


async def build(languages: list[str], fetch_web: bool) -> int:
    settings = get_settings()
    log.info(
        "Embeddings: %s/%s",
        settings.embedding_provider,
        providers.resolve_embedding_model(settings),
    )

    store, embeddings = _build_store()
    before = store.count()

    ingestion = IngestionPipeline(store, embeddings)
    try:
        await ingestion.run(languages)
    finally:
        await ingestion.close()

    after = store.count()
    log.info("Index: %d chunks (+%d)", after, after - before)
    if after == 0:
        log.error(
            "Index is empty. Check that backend/rag/reference/<lang>/ contains "
            "documents and that the embedding provider is reachable."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "languages",
        nargs="*",
        help="Languages to ingest. Default: every supported language.",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Skip fetching official docs; index only the local corpus.",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Report index size and exit."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
    )

    settings = get_settings()
    if args.stats:
        store, _ = _build_store()
        count = store.count()
        print(f"Chroma index: {count} chunks")
        # An empty index is the failure this script exists to prevent, and it is
        # invisible at runtime — retrieval just quietly returns nothing.
        return 0 if count else 1

    languages = args.languages or [
        lang["id"] for lang in settings.supported_languages
    ]
    if args.no_web:
        # Mutating the cached settings object is fine for a one-shot script and
        # keeps the flag from having to thread through the ingestion pipeline.
        settings.enable_web_docs = False

    log.info("Building index for: %s", ", ".join(languages))
    return asyncio.run(build(languages, not args.no_web))


if __name__ == "__main__":
    raise SystemExit(main())
