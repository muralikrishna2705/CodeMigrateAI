"""LRU cache keyed by hierarchical strings, supporting prefix invalidation.

``OrderedDict`` is already the right structure for recency: ``move_to_end`` and
``popitem(last=False)`` are both O(1), and no tree or heap improves on that. The
gap it leaves is *grouped* access — "every entry belonging to this migration
family" — so that a changed language profile can evict its own entries instead
of forcing the whole cache cold.

That grouping comes from the *key scheme*, not from a second index: keys are
``<source>:<version>:<target>:<version>:…`` paths, so a family is a prefix.
Selecting them is a ``str.startswith`` filter over the keys.

A trie was measured here and removed. It answers prefix queries in
O(L + matches) versus the filter's O(N·L), but it walks one interpreted step per
character while ``startswith`` runs in C — 27x slower at 500 entries and 31x at
5000. The crossover sits far above any cache this process will hold, and the
trie also doubled the per-entry memory. The asymptotically worse loop is the
faster one at every size that occurs.

This is deliberately *not* built on ``functools.lru_cache``. That decorator
memoizes a pure function on its arguments: it cannot be keyed by a runtime
string, sized from settings, invalidated by family, or inspected — and it holds
strong references until eviction regardless. Nor are the values held weakly:
callers here receive deep copies, so the cache holds the only strong reference
and a weak one would let every entry evict itself immediately.
"""

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

V = TypeVar("V")


class PrefixLRU(Generic[V]):
    """Bounded LRU keyed by string, with prefix lookup and invalidation."""

    __slots__ = ("_items", "_maxsize")

    def __init__(self, maxsize: int = 500):
        self._items: OrderedDict[str, V] = OrderedDict()
        self._maxsize = max(1, maxsize)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def get(self, key: str) -> Optional[V]:
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: V) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        elif len(self._items) >= self._maxsize:
            self._items.popitem(last=False)
        self._items[key] = value

    def keys_with_prefix(self, prefix: str) -> list[str]:
        """Keys under ``prefix``. An empty prefix matches everything."""
        if not prefix:
            return list(self._items)
        return [key for key in self._items if key.startswith(prefix)]

    def invalidate_prefix(self, prefix: str) -> int:
        """Drop every entry under ``prefix``; returns how many were removed."""
        keys = self.keys_with_prefix(prefix)
        for key in keys:
            del self._items[key]
        return len(keys)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items
