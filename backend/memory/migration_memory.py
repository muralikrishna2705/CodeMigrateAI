"""MigrationMemory — domain-level recall of past migrations.

The lookup is two-stage, because the two stages are good at different things:

1. **Trie filter (O(k))** — walk the language-pair key ``"java>python"`` to the
   candidate set. This is a *correctness* filter, not an optimisation: a Python
   migration must never be grounded in a Go precedent, however similar the code
   looks. Cosine alone cannot promise that, and a pure SQL ``WHERE`` cannot
   answer the prefix queries ("everything migrating *out of* java") the trie
   gets for free.
2. **Cosine rank** — score the surviving candidates against the incoming source
   by token-frequency cosine, then blend in each record's stored quality score.

Stage 1 is what keeps stage 2 cheap: cosine is O(candidates), so restricting the
candidate set by pair means a store with a hundred thousand rows still only
scores the handful that could possibly apply.

When embeddings are wired, :class:`rag.migration_memory.SemanticMigrationMemory`
is consulted as a third leg and its hits are merged in by entry id — the two
stores share an id scheme precisely so this merge needs no join table. The
lexical cosine here is not trying to beat embeddings; it is the offline-safe
floor that keeps recall working when no embedding service is available.
"""

import logging
import math
import re
from collections import Counter
from typing import Iterable, Optional

from .memory_store import MemoryStore, hash_code, make_entry_id, pair_key

log = logging.getLogger("CodeMigrateAI.MigrationMemory")

_MAX_EXCERPT_CHARS = 1500
# Identifiers only. Punctuation and one-character names carry no signal about
# what a migration *did*, and letting them into the vector lets brace-heavy
# languages look similar to each other purely on syntax.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


class _TrieNode:
    __slots__ = ("children", "entry_ids")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        self.entry_ids: list[str] = []


class LanguagePairTrie:
    """Prefix index over ``"<source>><target>"`` keys.

    ``insert``/``search`` are O(k) in the key length — independent of how many
    migrations are stored, which is the point: the candidate filter must not get
    slower as memory grows.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def insert(self, key: str, entry_id: str) -> None:
        node = self._root
        for char in key:
            node = node.children.setdefault(char, _TrieNode())
        if entry_id not in node.entry_ids:
            node.entry_ids.append(entry_id)
            self._size += 1

    def search(self, key: str) -> list[str]:
        """Exact-key lookup: entry ids for precisely this language pair."""
        node = self._root
        for char in key:
            node = node.children.get(char)
            if node is None:
                return []
        return list(node.entry_ids)

    def search_prefix(self, prefix: str) -> list[str]:
        """All entry ids under ``prefix`` (e.g. ``"java>"`` -> every java source)."""
        node = self._root
        for char in prefix:
            node = node.children.get(char)
            if node is None:
                return []
        found: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            found.extend(current.entry_ids)
            stack.extend(current.children.values())
        return found

    def __len__(self) -> int:
        return self._size


class MigrationMemory:
    """Structured, trie-indexed recall over :class:`MemoryStore`."""

    def __init__(
        self,
        store: MemoryStore,
        semantic_memory=None,
        *,
        min_similarity: float = 0.25,
    ):
        self._store = store
        self._semantic = semantic_memory
        self._min_similarity = min_similarity
        self._trie = LanguagePairTrie()
        self._loaded = False

    # ---------------------------------------------------------------- lifecycle

    def initialize(self) -> "MigrationMemory":
        """Open the store and rebuild the trie from persisted rows.

        The trie is a derived index held in memory, so it must be rebuilt at
        startup — this is exactly the step that makes recall *cross-session*
        rather than merely cross-request.
        """
        self._store.initialize()
        self._rebuild_trie()
        self._loaded = True
        log.info("Migration memory ready (%d indexed entries)", len(self._trie))
        return self

    def _rebuild_trie(self) -> None:
        self._trie = LanguagePairTrie()
        for row in self._store.iter_migrations():
            self._trie.insert(row["lang_pair"], row["entry_id"])

    def attach_semantic(self, semantic_memory) -> None:
        """Wire the Chroma leg in after the fact.

        The SQL store comes up immediately at startup, but the embedding service
        behind ``SemanticMigrationMemory`` initialises in a background task that
        may take minutes or fail outright. Recall therefore starts SQL-only and
        gains the semantic leg when (or if) it arrives.
        """
        self._semantic = semantic_memory

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def trie(self) -> LanguagePairTrie:
        return self._trie

    # ------------------------------------------------------------------- write

    def remember(
        self,
        *,
        source_code: str,
        source_language: str,
        target_language: str,
        migrated_code: str,
        source_version: str = "",
        target_version: str = "",
        plan: str = "",
        score: float = 0.0,
        user_feedback: str = "",
        session_id: str = "",
    ) -> str:
        entry_id = make_entry_id(source_code, target_language, target_version)
        self._store.save_migration(
            entry_id=entry_id,
            source_language=source_language,
            source_version=source_version,
            target_language=target_language,
            target_version=target_version,
            source_code_hash=hash_code(source_code),
            source_excerpt=source_code[:_MAX_EXCERPT_CHARS],
            migrated_code=migrated_code,
            plan=plan,
            score=score,
            user_feedback=user_feedback,
            session_id=session_id,
        )
        self._trie.insert(pair_key(source_language, target_language), entry_id)
        return entry_id

    def record_failure(
        self,
        *,
        source_code: str,
        source_language: str,
        target_language: str,
        reason: str,
        details=None,
        session_id: str = "",
    ) -> None:
        self._store.save_failure(
            source_language=source_language,
            target_language=target_language,
            source_code_hash=hash_code(source_code),
            reason=reason,
            details=details,
            session_id=session_id,
        )

    def record_feedback(
        self,
        entry_id: str,
        note: str,
        *,
        corrected_code: str = "",
        session_id: str = "",
    ) -> None:
        self._store.save_correction(
            entry_id=entry_id,
            corrected_code=corrected_code,
            note=note,
            session_id=session_id,
        )

    # -------------------------------------------------------------------- read

    def recall(
        self,
        *,
        source_code: str,
        source_language: str,
        target_language: str,
        k: int = 3,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Return up to ``k`` prior migrations most similar to ``source_code``.

        Each hit carries ``similarity`` (lexical cosine), ``score`` (the stored
        quality of that migration) and ``rank_score`` (the blend actually sorted
        on), so a caller can tell "very similar but scored badly" apart from
        "less similar but known good" instead of trusting one opaque number.
        """
        if not self._loaded:
            self.initialize()

        # Stage 1 — trie narrows to this language pair in O(k).
        candidate_ids = self._trie.search(pair_key(source_language, target_language))
        if not candidate_ids:
            return self._semantic_only(
                source_code, source_language, target_language, k
            )

        # An exact re-run of code we have seen before short-circuits the ranking.
        exact = self._store.find_by_hash(hash_code(source_code))
        exact_ids = {row["entry_id"] for row in exact}

        # Stage 2 — cosine over the survivors.
        query_vector = _vectorize(source_code)
        hits: list[dict] = []
        for entry_id in candidate_ids:
            row = self._store.get_migration(entry_id)
            if not row or row["score"] < min_score:
                continue
            similarity = (
                1.0
                if entry_id in exact_ids
                else _cosine(query_vector, _vectorize(row["source_excerpt"]))
            )
            if similarity < self._min_similarity:
                continue
            hits.append(
                {
                    "entry_id": entry_id,
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                    "target_version": row["target_version"],
                    "migrated_code": row["migrated_code"],
                    "plan": row["plan"],
                    "score": row["score"],
                    "user_feedback": row["user_feedback"],
                    "similarity": round(similarity, 4),
                    # Similarity dominates; the stored score only breaks ties
                    # between comparably-similar precedents. Weighting quality
                    # any higher would surface a well-scored but unrelated
                    # migration ahead of the one that actually matches.
                    "rank_score": round(similarity + 0.15 * row["score"], 4),
                    "origin": "sqlite",
                }
            )

        hits.sort(key=lambda h: h["rank_score"], reverse=True)
        merged = self._merge_semantic(
            hits, source_code, source_language, target_language, k
        )
        return merged[:k]

    def warnings_for(self, source_language: str, target_language: str) -> list[dict]:
        """Recent failures for this pair — read to warn, never to ground."""
        return self._store.recent_failures(pair_key(source_language, target_language))

    # ---------------------------------------------------------- semantic bridge

    def _semantic_only(
        self, source_code: str, source_language: str, target_language: str, k: int
    ) -> list[dict]:
        if self._semantic is None:
            return []
        return self._search_semantic(source_code, source_language, target_language, k)

    def _merge_semantic(
        self,
        hits: list[dict],
        source_code: str,
        source_language: str,
        target_language: str,
        k: int,
    ) -> list[dict]:
        if self._semantic is None:
            return hits
        seen = {h["entry_id"] for h in hits}
        for hit in self._search_semantic(
            source_code, source_language, target_language, k
        ):
            if hit["entry_id"] in seen:
                continue
            hits.append(hit)
        hits.sort(key=lambda h: h["rank_score"], reverse=True)
        return hits

    def _search_semantic(
        self, source_code: str, source_language: str, target_language: str, k: int
    ) -> list[dict]:
        try:
            results = self._semantic.search(
                source_code[:_MAX_EXCERPT_CHARS],
                k=k,
                where={"language": target_language},
            )
        except Exception as exc:  # noqa: BLE001 — semantic leg is optional
            log.debug("Semantic recall unavailable: %s", exc)
            return []

        hits = []
        for document, relevance in results:
            metadata = document.metadata or {}
            hits.append(
                {
                    "entry_id": metadata.get("entry_id", ""),
                    "source_language": metadata.get("source_language", source_language),
                    "target_language": metadata.get("language", target_language),
                    "target_version": metadata.get("version", ""),
                    "migrated_code": metadata.get("migrated_code", ""),
                    "plan": metadata.get("plan_summary", ""),
                    "score": 0.0,
                    "user_feedback": "",
                    "similarity": round(float(relevance), 4),
                    "rank_score": round(float(relevance), 4),
                    "origin": "semantic",
                }
            )
        return hits


# ------------------------------------------------------------ similarity utils


def _vectorize(text: str) -> Counter:
    return Counter(_TOKEN_RE.findall((text or "").lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller vector: the intersection is all that contributes.
    if len(b) < len(a):
        a, b = b, a
    dot = sum(count * b[token] for token, count in a.items() if token in b)
    if not dot:
        return 0.0
    norm_a = math.sqrt(sum(c * c for c in a.values()))
    norm_b = math.sqrt(sum(c * c for c in b.values()))
    return dot / (norm_a * norm_b)


def cosine_similarity(left: str, right: str) -> float:
    """Token-frequency cosine between two code excerpts, in [0.0, 1.0]."""
    return _cosine(_vectorize(left), _vectorize(right))


def summarize_hits(hits: Iterable[dict], limit: int = 3) -> str:
    """Render recall hits as prompt-ready context."""
    lines: list[str] = []
    for index, hit in enumerate(list(hits)[:limit], start=1):
        header = (
            f"### Prior migration {index} "
            f"({hit.get('source_language')} -> {hit.get('target_language')}"
            f"{' ' + hit['target_version'] if hit.get('target_version') else ''}, "
            f"similarity {hit.get('similarity', 0):.2f})"
        )
        lines.append(header)
        if hit.get("plan"):
            lines.append(f"Plan: {hit['plan'][:400]}")
        if hit.get("user_feedback"):
            # Surfaced prominently: a human said this, and it is the only signal
            # here that did not come from the system grading its own work.
            lines.append(f"User correction: {hit['user_feedback'][:400]}")
        if hit.get("migrated_code"):
            lines.append(f"```\n{hit['migrated_code'][:800]}\n```")
    return "\n".join(lines)


def build_memory(
    db_path: str, semantic_memory=None, min_similarity: float = 0.25
) -> Optional[MigrationMemory]:
    """Construct and initialize a MigrationMemory, or None if unavailable.

    Memory is an optimisation — a broken or unwritable database must degrade the
    system to "no recall", never take a migration down with it.
    """
    try:
        store = MemoryStore(db_path)
        return MigrationMemory(
            store, semantic_memory, min_similarity=min_similarity
        ).initialize()
    except Exception as exc:  # noqa: BLE001
        log.warning("Persistent memory unavailable (%s); continuing without it", exc)
        return None
