"""Cross-session persistence (Dimension 5).

Three layers, each answering a different question about the past:

``MemoryStore``
    The system of record. A SQLite database holding every migration, reusable
    pattern, failure and user correction. Structured columns mean exact filters
    ("what did we do for java->python at score >= 0.8") that a vector store
    cannot express.

``MigrationMemory``
    The domain facade the pipeline actually calls. Narrows candidates by
    language pair through a trie in O(k), then ranks the survivors by cosine
    similarity — and folds in the semantic leg
    (:class:`rag.migration_memory.SemanticMigrationMemory`) when embeddings are
    wired.

``PatternStore``
    Deduplicated conversion patterns, fronted by a bloom filter so the common
    "have we seen this already?" case answers in O(1) without touching SQLite.

Graph checkpointing (``build_checkpointer``) is separate: it persists *in-run*
graph state per ``thread_id`` so an interrupted migration can resume, whereas
the stores above persist *outcomes* across unrelated runs.
"""

from .checkpointer import build_checkpointer
from .memory_store import MemoryStore
from .migration_memory import LanguagePairTrie, MigrationMemory
from .pattern_store import BloomFilter, PatternStore

__all__ = [
    "MemoryStore",
    "MigrationMemory",
    "LanguagePairTrie",
    "PatternStore",
    "BloomFilter",
    "build_checkpointer",
]
