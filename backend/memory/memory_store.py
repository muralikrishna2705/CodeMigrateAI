"""MemoryStore — the SQLite system of record for everything a run learns.

Four tables, because the four things are read back differently:

``migrations``  completed runs, keyed by (language pair, source hash)
``patterns``    reusable conversion pairs, deduplicated by PatternStore
``failures``    runs that errored or failed validation
``corrections`` user edits to migrated code — the highest-value signal here,
                since it is the only row a human wrote

Failures are recorded even though :class:`rag.migration_memory.SemanticMigrationMemory`
deliberately refuses them. The distinction is how each is read: the semantic
store is *retrieved into prompts*, so a bad precedent there would launder a
failure into grounding. These rows are never injected as prior art — they are
read to warn ("this construct failed last time"), which needs the failures to
be on record.

stdlib ``sqlite3`` only: no server, no external dependency, and the file is
portable. Every method is synchronous and blocking; async callers reach it
through ``asyncio.to_thread`` like the other stores in this codebase.
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("CodeMigrateAI.MemoryStore")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS migrations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id          TEXT    NOT NULL UNIQUE,
    session_id        TEXT    NOT NULL DEFAULT '',
    source_language   TEXT    NOT NULL,
    source_version    TEXT    NOT NULL DEFAULT '',
    target_language   TEXT    NOT NULL,
    target_version    TEXT    NOT NULL DEFAULT '',
    lang_pair         TEXT    NOT NULL,
    source_code_hash  TEXT    NOT NULL,
    source_excerpt    TEXT    NOT NULL DEFAULT '',
    migrated_code     TEXT    NOT NULL DEFAULT '',
    plan              TEXT    NOT NULL DEFAULT '',
    score             REAL    NOT NULL DEFAULT 0.0,
    user_feedback     TEXT    NOT NULL DEFAULT '',
    created_at        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_migrations_pair  ON migrations (lang_pair);
CREATE INDEX IF NOT EXISTS idx_migrations_hash  ON migrations (source_code_hash);
CREATE INDEX IF NOT EXISTS idx_migrations_score ON migrations (score DESC);

CREATE TABLE IF NOT EXISTS patterns (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_hash   TEXT    NOT NULL UNIQUE,
    lang_pair      TEXT    NOT NULL,
    source_snippet TEXT    NOT NULL,
    target_snippet TEXT    NOT NULL,
    occurrences    INTEGER NOT NULL DEFAULT 1,
    score          REAL    NOT NULL DEFAULT 0.0,
    created_at     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_pair ON patterns (lang_pair);

CREATE TABLE IF NOT EXISTS failures (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT NOT NULL DEFAULT '',
    lang_pair        TEXT NOT NULL,
    source_code_hash TEXT NOT NULL,
    reason           TEXT NOT NULL DEFAULT '',
    details          TEXT NOT NULL DEFAULT '',
    created_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_pair ON failures (lang_pair);

CREATE TABLE IF NOT EXISTS corrections (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id       TEXT NOT NULL DEFAULT '',
    session_id     TEXT NOT NULL DEFAULT '',
    lang_pair      TEXT NOT NULL DEFAULT '',
    original_code  TEXT NOT NULL DEFAULT '',
    corrected_code TEXT NOT NULL DEFAULT '',
    note           TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_entry ON corrections (entry_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB
);
"""


class MemoryStore:
    """SQLite persistence for migrations, patterns, failures and corrections.

    Also exposes the LangChain ``BaseChatMemory`` surface
    (``memory_variables`` / ``load_memory_variables`` / ``save_context`` /
    ``clear``) so this can be dropped into a LangChain chain. It deliberately
    does not *subclass* ``BaseChatMemory``: that base models a turn-by-turn chat
    transcript on a ``ChatMessageHistory``, while these rows are structured
    migration records with scores and feedback. Inheriting would force the rows
    through a message log and lose exactly the columns that make them queryable.
    """

    def __init__(self, db_path: str | Path, *, timeout: float = 10.0):
        self.db_path = resolve_data_path(db_path)
        self._timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        # sqlite3 serialises writes itself, but the graph reaches this store from
        # worker threads (asyncio.to_thread) during the parallel fan-out, so a
        # single shared connection still needs guarding on our side.
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- lifecycle

    def initialize(self) -> "MemoryStore":
        with self._lock:
            if self._conn is not None:
                return self
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path), timeout=self._timeout, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            # WAL lets the read path (recall) proceed while a write is in flight;
            # the default rollback journal would block readers on every remember().
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        log.info(
            "Memory store ready at %s (%d migrations)",
            self.db_path,
            self.count("migrations"),
        )
        return self

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.initialize()
        return self._conn  # type: ignore[return-value]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------ writes

    def save_migration(
        self,
        *,
        entry_id: str,
        source_language: str,
        target_language: str,
        source_code_hash: str,
        migrated_code: str = "",
        plan: str = "",
        score: float = 0.0,
        user_feedback: str = "",
        source_version: str = "",
        target_version: str = "",
        source_excerpt: str = "",
        session_id: str = "",
    ) -> str:
        """Upsert one migration record. Returns the entry id.

        Re-running the same source against the same target overwrites rather
        than accumulating near-duplicates, matching the semantic store's
        behaviour so the two never disagree about how many times a migration
        "happened". User feedback survives the overwrite — it was written by a
        human about this code, and a later automated re-run has no grounds to
        discard it.
        """
        lang_pair = pair_key(source_language, target_language)
        now = time.time()
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO migrations (
                    entry_id, session_id, source_language, source_version,
                    target_language, target_version, lang_pair, source_code_hash,
                    source_excerpt, migrated_code, plan, score, user_feedback,
                    created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    migrated_code = excluded.migrated_code,
                    plan          = excluded.plan,
                    score         = excluded.score,
                    session_id    = excluded.session_id,
                    source_excerpt= excluded.source_excerpt,
                    user_feedback = CASE
                        WHEN excluded.user_feedback != '' THEN excluded.user_feedback
                        ELSE migrations.user_feedback END
                """,
                (
                    entry_id,
                    session_id,
                    source_language,
                    source_version,
                    target_language,
                    target_version,
                    lang_pair,
                    source_code_hash,
                    source_excerpt,
                    migrated_code,
                    plan,
                    float(score),
                    user_feedback,
                    now,
                ),
            )
            self.conn.commit()
        return entry_id

    def save_failure(
        self,
        *,
        source_language: str,
        target_language: str,
        source_code_hash: str,
        reason: str,
        details: Any = None,
        session_id: str = "",
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO failures
                   (session_id, lang_pair, source_code_hash, reason, details, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id,
                    pair_key(source_language, target_language),
                    source_code_hash,
                    reason[:2000],
                    _dump(details),
                    time.time(),
                ),
            )
            self.conn.commit()

    def save_correction(
        self,
        *,
        entry_id: str = "",
        original_code: str = "",
        corrected_code: str = "",
        note: str = "",
        source_language: str = "",
        target_language: str = "",
        session_id: str = "",
    ) -> None:
        """Record a human edit, and fold the note back onto its migration row.

        Writing the feedback onto ``migrations`` too is what makes corrections
        actually change future behaviour: ``recall`` ranks on that table, so a
        correction that only lived in its own table would never be seen by the
        lookup it is supposed to inform.
        """
        with self._lock:
            self.conn.execute(
                """INSERT INTO corrections
                   (entry_id, session_id, lang_pair, original_code, corrected_code,
                    note, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    entry_id,
                    session_id,
                    pair_key(source_language, target_language)
                    if source_language
                    else "",
                    original_code,
                    corrected_code,
                    note,
                    time.time(),
                ),
            )
            if entry_id:
                self.conn.execute(
                    """UPDATE migrations
                       SET user_feedback = ?, migrated_code = CASE
                           WHEN ? != '' THEN ? ELSE migrated_code END
                       WHERE entry_id = ?""",
                    (note, corrected_code, corrected_code, entry_id),
                )
            self.conn.commit()

    def set_score(self, entry_id: str, score: float) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE migrations SET score = ? WHERE entry_id = ?",
                (float(score), entry_id),
            )
            self.conn.commit()

    # ------------------------------------------------------------------- reads

    def get_migration(self, entry_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM migrations WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_by_hash(self, source_code_hash: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM migrations WHERE source_code_hash = ? ORDER BY score DESC",
            (source_code_hash,),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_pair(
        self, lang_pair: str, *, limit: int = 50, min_score: float = 0.0
    ) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM migrations
               WHERE lang_pair = ? AND score >= ?
               ORDER BY score DESC, created_at DESC LIMIT ?""",
            (lang_pair, min_score, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_pairs(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT lang_pair FROM migrations"
        ).fetchall()
        return [r["lang_pair"] for r in rows]

    def iter_migrations(self, limit: int = 10_000) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM migrations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_failures(self, lang_pair: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM failures WHERE lang_pair = ?
               ORDER BY created_at DESC LIMIT ?""",
            (lang_pair, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self, table: str = "migrations") -> int:
        if table not in {"migrations", "patterns", "failures", "corrections"}:
            raise ValueError(f"unknown table: {table}")
        try:
            # Table name is whitelisted above; sqlite cannot parameterise it.
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            return int(row["n"])
        except sqlite3.Error:  # counting is diagnostic only
            return 0

    # -------------------------------------------------------------- meta blobs

    def get_meta(self, key: str) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: bytes) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO meta (key, value) VALUES (?,?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )
            self.conn.commit()

    # ------------------------------------------- LangChain BaseChatMemory shape

    @property
    def memory_variables(self) -> list[str]:
        return ["migration_history"]

    def load_memory_variables(self, inputs: dict) -> dict:
        """Return prior migrations for the language pair named in ``inputs``."""
        lang_pair = inputs.get("lang_pair") or pair_key(
            inputs.get("source_language", ""), inputs.get("target_language", "")
        )
        return {
            "migration_history": self.find_by_pair(
                lang_pair, limit=int(inputs.get("k", 5))
            )
        }

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Persist one migration from a chain's inputs/outputs."""
        source_code = inputs.get("source_code", "")
        self.save_migration(
            entry_id=outputs.get("entry_id")
            or make_entry_id(
                source_code,
                inputs.get("target_language", ""),
                inputs.get("target_version", ""),
            ),
            source_language=inputs.get("source_language", ""),
            source_version=inputs.get("source_version", ""),
            target_language=inputs.get("target_language", ""),
            target_version=inputs.get("target_version", ""),
            source_code_hash=hash_code(source_code),
            source_excerpt=source_code[:1500],
            migrated_code=outputs.get("migrated_code", ""),
            plan=outputs.get("plan", ""),
            score=float(outputs.get("score", 0.0) or 0.0),
            session_id=inputs.get("session_id", ""),
        )

    def clear(self) -> None:
        with self._lock:
            for table in ("migrations", "patterns", "failures", "corrections", "meta"):
                self.conn.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed names
            self.conn.commit()


# ----------------------------------------------------------------- module utils

# backend/ — the package root, mirroring how rag/ anchors its chroma_db.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_path(db_path: str | Path) -> Path:
    """Anchor a relative database path to the backend package root.

    Settings carry paths like ``"memory/codemigrate.db"``. Resolved against the
    process CWD, that silently points at a *different* file depending on whether
    uvicorn was started from the repo root or from ``backend/`` — so a migration
    remembered in development would be invisible in production. Anchoring to the
    package makes the location a property of the install, not of the shell.

    Absolute paths are honoured as given, which is what tests and deployments
    that mount a volume need.
    """
    path = Path(db_path)
    return path if path.is_absolute() else _BACKEND_ROOT / path


def pair_key(source_language: str, target_language: str) -> str:
    """Trie key for a language pair. Lowercased so 'Java' and 'java' collide."""
    return f"{(source_language or '').lower()}>{(target_language or '').lower()}"


def hash_code(source_code: str) -> str:
    import hashlib

    return hashlib.sha256((source_code or "").encode()).hexdigest()


def make_entry_id(source_code: str, target_language: str, target_version: str) -> str:
    """Stable id for a (code, target) pair.

    Mirrors ``SemanticMigrationMemory.entry_id`` on purpose: the same migration
    gets the same id in both stores, so a SQLite row and its Chroma document can
    be joined without a second lookup table.
    """
    import hashlib

    digest = hashlib.sha256(
        f"{target_language}\n{target_version}\n{source_code}".encode()
    ).hexdigest()
    return f"mem-{digest[:32]}"


def _dump(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:4000]
    try:
        return json.dumps(value, default=str)[:4000]
    except (TypeError, ValueError):
        return str(value)[:4000]
