"""Trie (prefix tree) — the shared prefix index.

Backs :class:`memory.migration_memory.LanguagePairTrie`, which narrows recall
candidates to one language pair before the cosine stage ever runs. That filter
is a correctness boundary, not just a speed-up: a Python migration must never be
grounded in a Go precedent, and the open-ended query it also needs — *every*
migration out of Java, ``"java>"`` — is the one shape a hash table cannot serve
without inspecting all N keys.

Scope note: a trie only repays its cost when the prefix queries are open-ended
and the index is large. Applying one to the grounding check and to the migration
caches was measured and reverted — walking a trie costs one interpreted step per
character, where ``str.find`` and ``str.startswith`` run in C, and at those call
sites (tens of lookups, hundreds of keys) the C loop won by 15x and 21x
respectively despite the worse asymptotics. Prefer a scan until the collection
is genuinely large.
"""

from typing import Any, Iterator, Optional


class _TrieNode:
    __slots__ = ("children", "key", "value", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        # The full key is stored on the terminal node so prefix enumeration can
        # yield keys without rebuilding them from the walked path.
        self.key: Optional[str] = None
        self.value: Any = None
        self.is_terminal: bool = False


class Trie:
    """Prefix tree mapping string keys to optional values.

    All operations are O(L) in the *key length*, independent of the number of
    stored keys — except the prefix enumerations, which are O(L + size of the
    matched subtree), i.e. proportional to the answer rather than to the index.
    """

    __slots__ = ("_root", "_size")

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def insert(self, key: str, value: Any = None) -> None:
        """Insert ``key``; re-inserting an existing key overwrites its value."""
        if not key:
            return
        node = self._root
        for char in key:
            child = node.children.get(char)
            if child is None:
                child = _TrieNode()
                node.children[char] = child
            node = child
        if not node.is_terminal:
            node.is_terminal = True
            node.key = key
            self._size += 1
        node.value = value

    def _walk(self, path: str) -> Optional[_TrieNode]:
        """Return the node reached by ``path``, or None if it falls off the trie."""
        node = self._root
        for char in path:
            node = node.children.get(char)
            if node is None:
                return None
        return node

    def __contains__(self, key: str) -> bool:
        node = self._walk(key)
        return node is not None and node.is_terminal

    def get(self, key: str, default: Any = None) -> Any:
        node = self._walk(key)
        if node is None or not node.is_terminal:
            return default
        return node.value

    def items_with_prefix(self, prefix: str) -> Iterator[tuple[str, Any]]:
        """Yield ``(key, value)`` for every stored key under ``prefix``.

        Iterative rather than recursive: keys here can be long (cache keys,
        namespace paths) and a per-character recursion depth would risk a
        RecursionError on inputs we do not control.
        """
        node = self._walk(prefix)
        if node is None:
            return
        stack = [node]
        while stack:
            current = stack.pop()
            if current.is_terminal:
                yield current.key, current.value  # type: ignore[misc]
            stack.extend(current.children.values())

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return [key for key, _ in self.items_with_prefix(prefix)]

    def __len__(self) -> int:
        return self._size
