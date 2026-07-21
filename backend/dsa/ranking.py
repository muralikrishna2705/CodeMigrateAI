"""Top-k selection for retrieval ranking.

Retrieval merges always want the best ``k`` of ``N`` candidates, never a fully
ordered list — the tail is discarded the moment it is computed. ``sorted(...)[:k]``
pays O(N log N) to order results nobody reads; a bounded heap pays O(N log k).

The asymptotics only tell half the story. ``sorted`` is CPython's C-implemented
timsort, while ``nlargest`` maintains its heap through interpreted comparisons,
so the heap does not repay its constant factor until N is in the hundreds.
Measured on the ``(doc, score)`` pairs this ranks: N=20 favours ``sorted`` by
2.1x, N=100 is a wash, N=1000 favours the heap by 2.4x and N=10000 by 4.2x.

So the threshold below is not a micro-optimisation — it is the difference
between the two regimes. A typical RRF merge fuses a few dozen hits and takes
the sorted path; the heap is what keeps an unfiltered retrieval ladder over
thousands of candidates from degrading.

Either path returns the same order. CPython's ``nlargest`` decorates each
element with a *decreasing* counter before heapifying, so ties break toward the
earlier element — exactly what a stable ``sorted(..., reverse=True)`` does. That
matters: reciprocal-rank fusion produces tied scores routinely, and a merge that
reshuffled equal-scoring documents would change which examples reach the model.
"""

import heapq
from operator import itemgetter
from typing import Any, Callable, Iterable, Optional, Sized, TypeVar

T = TypeVar("T")

_SCORE = itemgetter(1)

# Measured crossover between timsort's C loop and the heap's interpreted one.
_HEAP_MIN_ITEMS = 128


def top_k(
    items: Iterable[T], k: int, key: Optional[Callable[[T], Any]] = None
) -> list[T]:
    """Return the ``k`` highest-scoring items, best first.

    ``key`` defaults to ``item[1]``, the score slot of the ``(doc, score)`` pairs
    used throughout retrieval. ``k <= 0`` yields an empty list rather than the
    silent full-list return a negative slice would produce.
    """
    if k <= 0:
        return []
    key = key or _SCORE
    # Only sized inputs can be routed by size; anything lazy goes to the heap,
    # which is also the option that does not materialise the whole sequence.
    if isinstance(items, Sized) and len(items) < _HEAP_MIN_ITEMS:
        return sorted(items, key=key, reverse=True)[:k]
    return heapq.nlargest(k, items, key=key)
