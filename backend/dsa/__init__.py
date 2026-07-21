"""Shared data structures (Dimension 6).

``Trie``
    Prefix index over language-pair keys, backing the migration-memory candidate
    filter — the one call site whose queries are open-ended (*every* migration
    out of Java) rather than exact.

``PrefixLRU``
    ``OrderedDict`` recency (already O(1)) plus invalidation by key prefix, so a
    changed language profile evicts its own entries instead of the whole cache.

``top_k``
    Best-k selection that routes between ``sorted`` and a bounded heap at the
    measured crossover, so neither regime pays the other's constant factor.

A note on what is *not* here, since each was tried and measured. Tries at the
grounding and cache call sites lost to ``str.find``/``str.startswith`` by 15x
and 21x: an interpreted step per character cannot beat a C loop until the
collection is far larger than these hold. A bloom filter in front of an
in-memory dict costs more to build than the lookups it saves — the one in
:mod:`memory.pattern_store` is worth it only because it fronts a SQLite round
trip. And a pure-Python FNV-1a hash is 2.7x slower than ``hashlib.sha256`` on
short strings and 183x on 32KB, because SHA-256 is C and the FNV loop is not;
the cache keys that used SHA-256 now use no hash at all.
"""

from .lru import PrefixLRU
from .ranking import top_k
from .trie import Trie

__all__ = ["Trie", "PrefixLRU", "top_k"]
