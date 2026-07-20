"""PatternStore — deduplicated conversion patterns, fronted by a bloom filter.

A migration run proposes the same conversions over and over ("``ArrayList<T>``
becomes ``list[T]``"). Writing each occurrence would bloat the table and skew
any later frequency ranking, so every candidate is checked for membership first.

The bloom filter makes that check O(1) in *time* and, more importantly, O(m) in
*space* regardless of how many patterns are stored — a few KB of bits answers
"definitely new" for the overwhelming majority of candidates without a SQLite
round trip. Bloom filters are one-sided: a "not present" answer is exact, a
"present" answer may be a false positive. That asymmetry is the right way round
here, since we confirm every claimed hit against SQLite before treating it as a
duplicate. The filter never causes a wrong answer — only an occasional wasted
lookup.

The bit array is persisted to the ``meta`` table so the filter survives restarts;
losing it would silently re-admit every duplicate after a reboot.
"""

import hashlib
import logging
import math
import time
from typing import Optional

from .memory_store import MemoryStore, pair_key

log = logging.getLogger("CodeMigrateAI.PatternStore")

_BLOOM_META_KEY = "pattern_bloom_v1"


class BloomFilter:
    """Space-efficient probabilistic set: no false negatives, tunable false positives."""

    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < error_rate < 1.0:
            raise ValueError("error_rate must be in (0, 1)")
        self.capacity = capacity
        self.error_rate = error_rate
        # Standard sizing: m = -n·ln(p)/(ln2)², k = (m/n)·ln2.
        self.num_bits = max(8, int(-capacity * math.log(error_rate) / (math.log(2) ** 2)))
        self.num_hashes = max(1, int(round((self.num_bits / capacity) * math.log(2))))
        self.num_bits = 8 * ((self.num_bits + 7) // 8)  # whole bytes
        self._bits = bytearray(self.num_bits // 8)
        self.count = 0

    def _offsets(self, item: str):
        """Kirsch-Mitzenmacher double hashing: k indices from one digest.

        Two independent hashes combined as ``h1 + i*h2`` give the same false
        positive rate as k independent hash functions, for one hash computation.
        """
        digest = hashlib.sha256(item.encode()).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") | 1  # odd -> coprime with 2^n
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    def add(self, item: str) -> None:
        for offset in self._offsets(item):
            self._bits[offset >> 3] |= 1 << (offset & 7)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            self._bits[offset >> 3] & (1 << (offset & 7))
            for offset in self._offsets(item)
        )

    def to_bytes(self) -> bytes:
        header = (
            f"{self.capacity}|{self.error_rate}|{self.num_bits}|"
            f"{self.num_hashes}|{self.count}\n"
        ).encode()
        return header + bytes(self._bits)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "BloomFilter":
        header, _, bits = blob.partition(b"\n")
        capacity, error_rate, num_bits, num_hashes, count = header.decode().split("|")
        instance = cls(int(capacity), float(error_rate))
        instance.num_bits = int(num_bits)
        instance.num_hashes = int(num_hashes)
        instance.count = int(count)
        instance._bits = bytearray(bits)
        return instance

    @property
    def saturation(self) -> float:
        """Fraction of capacity used. Past 1.0 the false positive rate degrades."""
        return self.count / self.capacity if self.capacity else 0.0


class PatternStore:
    """Successful conversion patterns per language pair, deduplicated on write."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        capacity: int = 10_000,
        error_rate: float = 0.01,
    ):
        self._store = store
        self._capacity = capacity
        self._error_rate = error_rate
        self._bloom: Optional[BloomFilter] = None

    def initialize(self) -> "PatternStore":
        self._store.initialize()
        blob = self._store.get_meta(_BLOOM_META_KEY)
        if blob:
            try:
                self._bloom = BloomFilter.from_bytes(blob)
            except Exception as exc:  # noqa: BLE001 — corrupt blob -> rebuild
                log.warning("Bloom filter unreadable (%s); rebuilding", exc)
        if self._bloom is None:
            self._bloom = BloomFilter(self._capacity, self._error_rate)
            self._reindex()
        log.info(
            "Pattern store ready (%d patterns, bloom %.0f%% saturated)",
            self._store.count("patterns"),
            self._bloom.saturation * 100,
        )
        return self

    def _reindex(self) -> None:
        rows = self._store.conn.execute("SELECT pattern_hash FROM patterns").fetchall()
        for row in rows:
            self._bloom.add(row["pattern_hash"])

    @property
    def bloom(self) -> BloomFilter:
        if self._bloom is None:
            self.initialize()
        return self._bloom  # type: ignore[return-value]

    @staticmethod
    def pattern_hash(lang_pair: str, source_snippet: str, target_snippet: str) -> str:
        # Whitespace-insensitive: the same conversion reformatted is not a new
        # pattern, and treating it as one is the main source of near-duplicates.
        normalized = " ".join(f"{lang_pair}|{source_snippet}|{target_snippet}".split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def add_pattern(
        self,
        *,
        source_language: str,
        target_language: str,
        source_snippet: str,
        target_snippet: str,
        score: float = 0.0,
    ) -> bool:
        """Store one pattern. Returns True if newly added, False if a duplicate.

        A bloom hit is treated as *suspected* duplicate and confirmed against
        SQLite, so a false positive costs one query rather than a lost pattern.
        """
        lang_pair = pair_key(source_language, target_language)
        digest = self.pattern_hash(lang_pair, source_snippet, target_snippet)

        if digest in self.bloom:
            row = self._store.conn.execute(
                "SELECT id FROM patterns WHERE pattern_hash = ?", (digest,)
            ).fetchone()
            if row is not None:
                self._store.conn.execute(
                    "UPDATE patterns SET occurrences = occurrences + 1 WHERE id = ?",
                    (row["id"],),
                )
                self._store.conn.commit()
                return False
            # False positive — fall through and insert.

        self._store.conn.execute(
            """INSERT OR IGNORE INTO patterns
               (pattern_hash, lang_pair, source_snippet, target_snippet,
                occurrences, score, created_at)
               VALUES (?,?,?,?,1,?,?)""",
            (
                digest,
                lang_pair,
                source_snippet[:2000],
                target_snippet[:2000],
                float(score),
                time.time(),
            ),
        )
        self._store.conn.commit()
        self.bloom.add(digest)
        self._persist_bloom()
        return True

    def get_patterns(
        self, source_language: str, target_language: str, limit: int = 10
    ) -> list[dict]:
        """Most-repeated patterns for a pair — frequency is the quality signal."""
        rows = self._store.conn.execute(
            """SELECT * FROM patterns WHERE lang_pair = ?
               ORDER BY occurrences DESC, score DESC LIMIT ?""",
            (pair_key(source_language, target_language), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def seen(self, source_language: str, target_language: str, source_snippet: str,
             target_snippet: str) -> bool:
        """O(1) membership probe. False is exact; True may be a false positive."""
        digest = self.pattern_hash(
            pair_key(source_language, target_language), source_snippet, target_snippet
        )
        return digest in self.bloom

    def _persist_bloom(self) -> None:
        try:
            self._store.set_meta(_BLOOM_META_KEY, self.bloom.to_bytes())
        except Exception as exc:  # noqa: BLE001 — filter rebuilds from SQLite
            log.debug("Could not persist bloom filter: %s", exc)
