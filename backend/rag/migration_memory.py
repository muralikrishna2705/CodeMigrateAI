"""SemanticMigrationMemory — past migrations, searchable by meaning.

This is the *semantic leg* of persistent memory and the store behind
``SemanticSearchTool``. The structured system of record lives in
:mod:`memory.memory_store`; :class:`memory.migration_memory.MigrationMemory` is
the facade that queries both and merges their hits by entry id. The two share
the ``mem-<sha256>`` id scheme so that merge needs no join table.

Kept separate from the SQL store rather than folded into it because they fail
differently: this one needs a live embedding service and returns fuzzy matches,
while SQLite is always available and returns exact ones. Collapsing them would
make all recall depend on embeddings being up.

The reference corpus
(``VectorStore``, collection ``codemigrate_ref``) holds *documentation*: how the
target language works in general. This holds *experience*: what this system
actually did on a previous migration and whether it validated cleanly. They
answer different questions, so they are separate collections rather than one —
mixing them would let a past (possibly wrong) migration outrank an official doc
during reference retrieval.

Entries are written at the end of a run by the ObserverAgent and survive process
restarts via Chroma's on-disk persistence, which is what makes the lookup
*cross-session*: a migration today can surface how the same construct was handled
last week.

Only successful, validated migrations are recorded. Remembering a failed attempt
would ground future runs in known-bad code — worse than having no memory at all.
"""

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

log = logging.getLogger("CodeMigrateAI.SemanticMigrationMemory")

PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "codemigrate_memory"

# Bound what a single memory entry embeds/stores. Memories are hints, not the
# source of truth, so a summary-sized excerpt is enough and keeps the collection
# from ballooning to the size of every migration ever run.
_MAX_SNIPPET_CHARS = 1500


class SemanticMigrationMemory:
    """Semantic store of completed migrations, keyed by the source code's shape."""

    def __init__(self, embedding_service):
        self._embeddings = embedding_service
        self._store: Chroma | None = None

    def initialize(self) -> None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(PERSIST_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        log.info("Migration memory initialized with %d entries", self.count())

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self.initialize()
        return self._store

    @staticmethod
    def entry_id(source_code: str, target_language: str, target_version: str) -> str:
        """Stable id for a (code, target) pair, so re-running overwrites rather
        than accumulating near-duplicate memories of the same migration."""
        digest = hashlib.sha256(
            f"{target_language}\n{target_version}\n{source_code}".encode()
        ).hexdigest()
        return f"mem-{digest[:32]}"

    def remember(
        self,
        source_code: str,
        source_language: str,
        source_version: str,
        target_language: str,
        target_version: str,
        migrated_code: str,
        plan_summary: str = "",
    ) -> str:
        """Record one completed migration. Returns the entry id."""
        # The embedded text is the SOURCE side plus the plan: a future migration
        # searches with code it is about to migrate, so what must match is "code
        # that looks like this", not the output it produced.
        content = "\n".join(
            filter(
                None,
                [
                    f"# {source_language} {source_version} -> "
                    f"{target_language} {target_version}",
                    plan_summary,
                    "## Source",
                    source_code[:_MAX_SNIPPET_CHARS],
                ],
            )
        )
        entry_id = self.entry_id(source_code, target_language, target_version)
        document = Document(
            page_content=content,
            metadata={
                # Carried in metadata so a semantic hit can be joined back to its
                # SQLite row (scores, user feedback) without a second lookup.
                "entry_id": entry_id,
                "source_language": source_language,
                "source_version": source_version,
                "language": target_language,
                "version": target_version,
                "doc_type": "past-migration",
                "plan_summary": plan_summary[:500],
                "migrated_code": migrated_code[:_MAX_SNIPPET_CHARS],
            },
        )
        self.store.add_documents([document], ids=[entry_id])
        log.info(
            "Remembered migration %s (%s -> %s %s)",
            entry_id,
            source_language,
            target_language,
            target_version,
        )
        return entry_id

    def search(
        self, query: str, k: int = 3, where: dict | None = None
    ) -> list[tuple[Document, float]]:
        kwargs = {"k": k}
        if where:
            kwargs["filter"] = where
        return self.store.similarity_search_with_relevance_scores(query, **kwargs)

    def count(self) -> int:
        try:
            return len(self.store.get()["ids"])
        except Exception:  # noqa: BLE001 — count is diagnostic only
            return 0
