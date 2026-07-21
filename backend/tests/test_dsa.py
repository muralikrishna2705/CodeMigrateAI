"""Tests for the shared data structures and the call sites built on them."""

import random

import pytest
from cache.keys import KEY_NAMESPACE, generate_key, key_prefix
from cache.manager import LRUCache
from dsa import PrefixLRU, Trie, top_k


class TestTrie:
    def test_insert_and_exact_membership(self):
        trie = Trie()
        trie.insert("java.util.List", 1)
        assert "java.util.List" in trie
        assert trie.get("java.util.List") == 1
        assert "java.util" not in trie
        assert trie.get("missing", "default") == "default"

    def test_len_counts_distinct_keys(self):
        trie = Trie()
        trie.insert("alpha")
        trie.insert("alpha")  # re-insert must not double count
        trie.insert("alphabet")
        assert len(trie) == 2

    def test_empty_key_is_ignored(self):
        trie = Trie()
        trie.insert("")
        assert len(trie) == 0

    def test_reinsert_overwrites_the_value(self):
        trie = Trie()
        trie.insert("k", "first")
        trie.insert("k", "second")
        assert trie.get("k") == "second"
        assert len(trie) == 1


class TestTriePrefixEnumeration:
    def test_keys_with_prefix(self):
        trie = Trie()
        for key in ("migrate:py:java:a", "migrate:py:go:b", "migrate:rs:java:c"):
            trie.insert(key)
        assert sorted(trie.keys_with_prefix("migrate:py:")) == [
            "migrate:py:go:b",
            "migrate:py:java:a",
        ]

    def test_empty_prefix_returns_everything(self):
        trie = Trie()
        trie.insert("a")
        trie.insert("b")
        assert sorted(trie.keys_with_prefix("")) == ["a", "b"]

    def test_unmatched_prefix_returns_nothing(self):
        trie = Trie()
        trie.insert("alpha")
        assert trie.keys_with_prefix("zzz") == []

    def test_values_come_back_with_keys(self):
        trie = Trie()
        trie.insert("x:1", "one")
        assert list(trie.items_with_prefix("x:")) == [("x:1", "one")]

    def test_deep_keys_do_not_blow_the_stack(self):
        # Enumeration is iterative: a key longer than the recursion limit must
        # still be walkable.
        trie = Trie()
        deep = "a" * 5000
        trie.insert(deep)
        assert trie.keys_with_prefix("a" * 10) == [deep]

    def test_multimap_values_survive_prefix_enumeration(self):
        # This is exactly how LanguagePairTrie stores and reads back entry ids.
        trie = Trie()
        trie.insert("java>python", ["a"])
        trie.insert("java>go", ["b", "c"])
        trie.insert("rust>go", ["d"])
        found = [i for _, ids in trie.items_with_prefix("java>") for i in ids]
        assert sorted(found) == ["a", "b", "c"]


class TestTopK:
    def test_returns_best_k_in_descending_order(self):
        items = [("a", 0.1), ("b", 0.9), ("c", 0.5)]
        assert top_k(items, 2) == [("b", 0.9), ("c", 0.5)]

    def test_non_positive_k_returns_empty(self):
        items = [("a", 0.1)]
        assert top_k(items, 0) == []
        assert top_k(items, -1) == []

    def test_k_larger_than_input_returns_all(self):
        items = [("a", 0.1), ("b", 0.2)]
        assert top_k(items, 10) == [("b", 0.2), ("a", 0.1)]

    def test_ties_break_exactly_as_a_stable_sort(self):
        # RRF produces tied scores constantly; a different tie order would
        # change which reference examples reach the model.
        rng = random.Random(1234)
        items = [(f"doc{i}", rng.choice([0.5, 0.7, 0.9])) for i in range(60)]
        for k in (1, 5, 25, 60):
            expected = sorted(items, key=lambda p: p[1], reverse=True)[:k]
            assert top_k(items, k) == expected

    def test_custom_key(self):
        items = [{"s": 3}, {"s": 9}]
        assert top_k(items, 1, key=lambda d: d["s"]) == [{"s": 9}]


class TestPrefixLRU:
    def test_evicts_least_recently_used(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert "a" not in cache
        assert "b" in cache and "c" in cache

    def test_get_promotes_recency(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # 'b' is now the coldest
        cache.set("c", 3)
        assert "a" in cache
        assert "b" not in cache

    def test_overwrite_does_not_evict(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 99)
        assert len(cache) == 2
        assert cache.get("a") == 99

    def test_eviction_keeps_the_index_in_sync(self):
        # A stale trie entry would make keys_with_prefix report a key the cache
        # can no longer return.
        cache: PrefixLRU[int] = PrefixLRU(maxsize=2)
        cache.set("p:a", 1)
        cache.set("p:b", 2)
        cache.set("p:c", 3)
        assert sorted(cache.keys_with_prefix("p:")) == ["p:b", "p:c"]

    def test_invalidate_prefix_drops_only_that_family(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=10)
        cache.set("py:java:1", 1)
        cache.set("py:go:2", 2)
        cache.set("rs:java:3", 3)
        assert cache.invalidate_prefix("py:") == 2
        assert len(cache) == 1
        assert "rs:java:3" in cache

    def test_invalidate_empty_prefix_drops_all(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.invalidate_prefix("") == 2
        assert len(cache) == 0

    def test_maxsize_floor_is_one(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=0)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_missing_key_returns_none(self):
        cache: PrefixLRU[int] = PrefixLRU(maxsize=2)
        assert cache.get("nope") is None


class _FakeState:
    """Minimal stand-in for MigrationState — the cache only stores and returns it."""

    def __init__(self, tag: str):
        self.tag = tag


class TestCacheKeys:
    def test_key_is_hierarchical(self):
        state = _State("python", "3.8", "java", "21", "x = 1")
        key = generate_key(state)
        assert key.startswith(f"{KEY_NAMESPACE}:python:3.8:java:21:")
        # Namespace + four levels + digest.
        assert len(key.split(":")) == 6

    def test_same_inputs_give_the_same_key(self):
        a = _State("python", "3.8", "java", "21", "x = 1")
        b = _State("python", "3.8", "java", "21", "x = 1")
        assert generate_key(a) == generate_key(b)

    def test_different_code_gives_a_different_key(self):
        a = _State("python", "3.8", "java", "21", "x = 1")
        b = _State("python", "3.8", "java", "21", "x = 2")
        assert generate_key(a) != generate_key(b)

    def test_missing_version_still_fills_its_level(self):
        key = generate_key(_State("python", "", "java", "", "x = 1"))
        assert key.startswith(f"{KEY_NAMESPACE}:python:any:java:any:")
        assert len(key.split(":")) == 6

    def test_separators_in_a_component_cannot_forge_a_level(self):
        key = generate_key(_State("py:thon", "3.8", "java", "21", "x = 1"))
        assert len(key.split(":")) == 6

    def test_key_prefix_stops_at_the_first_omitted_level(self):
        assert key_prefix("python") == f"{KEY_NAMESPACE}:python:"
        assert key_prefix("python", "3.8") == f"{KEY_NAMESPACE}:python:3.8:"
        # A target cannot be pinned while the source version is open.
        assert key_prefix("python", "", "java") == f"{KEY_NAMESPACE}:python:"

    def test_generated_keys_live_under_their_prefix(self):
        key = generate_key(_State("Python", "3.8", "Java", "21", "x = 1"))
        assert key.startswith(key_prefix("python", "3.8", "java", "21"))


class _State:
    """Duck-typed MigrationState for key generation."""

    def __init__(self, src, src_v, tgt, tgt_v, code):
        self.source_language = src
        self.source_version = src_v
        self.target_language = tgt
        self.target_version = tgt_v
        self.source_code = code


class TestLRUCacheInvalidation:
    def test_invalidate_prefix_targets_one_family(self):
        cache = LRUCache(maxsize=10)
        py_java = generate_key(_State("python", "3.8", "java", "21", "a"))
        py_go = generate_key(_State("python", "3.8", "go", "1.22", "b"))
        rs_java = generate_key(_State("rust", "1.7", "java", "21", "c"))
        for key in (py_java, py_go, rs_java):
            cache.set(key, _FakeState(key))

        removed = cache.invalidate_prefix(key_prefix("python"))
        assert removed == 2
        assert cache.get(rs_java) is not None
        assert cache.get(py_java) is None

    def test_invalidate_can_narrow_to_an_exact_pair(self):
        cache = LRUCache(maxsize=10)
        py_java = generate_key(_State("python", "3.8", "java", "21", "a"))
        py_go = generate_key(_State("python", "3.8", "go", "1.22", "b"))
        cache.set(py_java, _FakeState("a"))
        cache.set(py_go, _FakeState("b"))

        assert cache.invalidate_prefix(key_prefix("python", "3.8", "java", "21")) == 1
        assert cache.get(py_go) is not None


class _StubSettings:
    redis_enabled = False
    redis_url = ""
    redis_ttl_seconds = 60
    local_cache_max_entries = 50


class TestCacheManagerInvalidate:
    """Scoped invalidation through the manager, with Redis disabled."""

    def _manager(self):
        from cache.manager import CacheManager

        return CacheManager(settings=_StubSettings())

    def test_invalidate_scopes_to_the_named_family(self):
        manager = self._manager()
        py_java = generate_key(_State("python", "3.8", "java", "21", "a"))
        go_java = generate_key(_State("go", "1.22", "java", "21", "b"))
        manager.local.set(py_java, _FakeState("a"))
        manager.local.set(go_java, _FakeState("b"))

        assert manager.invalidate("python") == 1
        assert manager.local.get(go_java) is not None
        assert manager.local.get(py_java) is None

    def test_invalidate_narrows_to_an_exact_target(self):
        manager = self._manager()
        to_java = generate_key(_State("python", "3.8", "java", "21", "a"))
        to_go = generate_key(_State("python", "3.8", "go", "1.22", "b"))
        manager.local.set(to_java, _FakeState("a"))
        manager.local.set(to_go, _FakeState("b"))

        assert manager.invalidate("python", "3.8", "java", "21") == 1
        assert manager.local.get(to_go) is not None

    def test_clear_still_empties_everything(self):
        manager = self._manager()
        manager.local.set(generate_key(_State("python", "3.8", "java", "21", "a")),
                          _FakeState("a"))
        manager.clear()
        assert len(manager.local) == 0


class TestRRFMergeOrdering:
    def test_merge_is_unchanged_by_heap_selection(self):
        from rag.retrieval_pipeline import RAGPipeline

        class _Doc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {}

        vector = [(_Doc(f"v{i}"), 0.9 - i * 0.01) for i in range(10)]
        keyword = [(_Doc(f"v{i}"), 0.5) for i in range(5, 15)]
        merged = RAGPipeline._rrf_merge(vector, keyword, k=5, rrf_k=60)

        assert len(merged) == 5
        # Documents in both legs fuse to the top; identity is page content, so
        # the shared ones are not duplicated.
        contents = [doc.page_content for doc, _ in merged]
        assert len(set(contents)) == 5

    def test_dedup_is_by_content_not_object_identity(self):
        from rag.retrieval_pipeline import merge_hits

        class _Doc:
            def __init__(self, content):
                self.page_content = content

        a = [(_Doc("same"), 0.4)]
        b = [(_Doc("same"), 0.8)]
        merged = merge_hits([a, b], k=5)
        assert len(merged) == 1
        assert merged[0][1] == 0.8  # keeps the best score


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
