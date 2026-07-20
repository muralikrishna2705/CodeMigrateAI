"""Tests for persistent memory + LangGraph checkpointing (Dimension 5).

Everything runs offline against temporary SQLite files — no embedding service,
no Ollama, no validator. The semantic leg is exercised through a stub so the
merge logic is covered without requiring Chroma.
"""

import asyncio

import pytest

from graph.migration_graph import build_migration_graph
from memory.checkpointer import build_checkpointer
from memory.memory_store import MemoryStore, hash_code, pair_key
from memory.migration_memory import (
    LanguagePairTrie,
    MigrationMemory,
    build_memory,
    cosine_similarity,
    summarize_hits,
)
from memory.pattern_store import BloomFilter, PatternStore
from models.state import MigrationState
from pipeline.orchestrator import Pipeline, _quality_score

JAVA_SRC = "public class Foo { List<String> items = new ArrayList<>(); }"
PY_SRC = "class Foo:\n    def __init__(self):\n        self.items: list[str] = []"


@pytest.fixture
def store(tmp_path):
    memory_store = MemoryStore(tmp_path / "test.db").initialize()
    yield memory_store
    memory_store.close()


@pytest.fixture
def memory(tmp_path):
    store = MemoryStore(tmp_path / "mem.db")
    yield MigrationMemory(store, min_similarity=0.1).initialize()
    store.close()


class TestMemoryStore:
    def test_save_and_read_back_a_migration(self, store):
        store.save_migration(
            entry_id="e1",
            source_language="java",
            target_language="python",
            source_code_hash=hash_code(JAVA_SRC),
            migrated_code=PY_SRC,
            plan="convert collections",
            score=0.9,
        )
        row = store.get_migration("e1")
        assert row["migrated_code"] == PY_SRC
        assert row["lang_pair"] == "java>python"
        assert row["score"] == 0.9

    def test_rerunning_a_migration_overwrites_rather_than_duplicates(self, store):
        for code in ("v1", "v2"):
            store.save_migration(
                entry_id="e1",
                source_language="java",
                target_language="python",
                source_code_hash="h",
                migrated_code=code,
            )
        assert store.count("migrations") == 1
        assert store.get_migration("e1")["migrated_code"] == "v2"

    def test_rerun_does_not_discard_human_feedback(self, store):
        # A human wrote this about this code; a later automated re-run has no
        # grounds to erase it.
        store.save_migration(
            entry_id="e1",
            source_language="java",
            target_language="python",
            source_code_hash="h",
            user_feedback="prefer pathlib",
        )
        store.save_migration(
            entry_id="e1",
            source_language="java",
            target_language="python",
            source_code_hash="h",
            migrated_code="new",
        )
        assert store.get_migration("e1")["user_feedback"] == "prefer pathlib"

    def test_correction_writes_back_onto_the_migration_row(self, store):
        # recall() ranks on `migrations`, so a correction that only lived in its
        # own table would never reach the lookup it is meant to inform.
        store.save_migration(
            entry_id="e1",
            source_language="java",
            target_language="python",
            source_code_hash="h",
            migrated_code="bad",
        )
        store.save_correction(
            entry_id="e1", corrected_code="good", note="use list comprehension"
        )
        row = store.get_migration("e1")
        assert row["migrated_code"] == "good"
        assert row["user_feedback"] == "use list comprehension"
        assert store.count("corrections") == 1

    def test_failures_are_recorded_separately_from_migrations(self, store):
        store.save_failure(
            source_language="java",
            target_language="python",
            source_code_hash="h",
            reason="syntax error",
            details={"line": 3},
        )
        assert store.count("failures") == 1
        assert store.count("migrations") == 0
        assert store.recent_failures("java>python")[0]["reason"] == "syntax error"

    def test_find_by_pair_filters_and_orders_by_score(self, store):
        for entry_id, target, score in (
            ("a", "python", 0.5),
            ("b", "python", 0.9),
            ("c", "go", 1.0),
        ):
            store.save_migration(
                entry_id=entry_id,
                source_language="java",
                target_language=target,
                source_code_hash="h",
                score=score,
            )
        rows = store.find_by_pair("java>python")
        assert [r["entry_id"] for r in rows] == ["b", "a"]

    def test_count_rejects_unknown_table_names(self, store):
        # The table name is interpolated into SQL, so the whitelist is load-bearing.
        with pytest.raises(ValueError):
            store.count("migrations; DROP TABLE migrations")

    def test_langchain_memory_surface_round_trips(self, store):
        store.save_context(
            {
                "source_code": JAVA_SRC,
                "source_language": "java",
                "target_language": "python",
            },
            {"migrated_code": PY_SRC, "score": 0.8},
        )
        loaded = store.load_memory_variables(
            {"source_language": "java", "target_language": "python"}
        )
        assert store.memory_variables == ["migration_history"]
        assert loaded["migration_history"][0]["migrated_code"] == PY_SRC

    def test_clear_empties_every_table(self, store):
        store.save_migration(
            entry_id="e1",
            source_language="java",
            target_language="python",
            source_code_hash="h",
        )
        store.clear()
        assert store.count("migrations") == 0


class TestDataPathResolution:
    def test_relative_paths_anchor_to_the_package_not_the_cwd(self, monkeypatch, tmp_path):
        # Otherwise the database depends on where uvicorn was launched, and a
        # migration remembered in dev would be invisible in production.
        from memory.memory_store import resolve_data_path

        monkeypatch.chdir(tmp_path)
        resolved = resolve_data_path("memory/codemigrate.db")
        assert resolved.is_absolute()
        assert resolved.parent.name == "memory"
        assert tmp_path not in resolved.parents

    def test_absolute_paths_are_honoured_as_given(self, tmp_path):
        from memory.memory_store import resolve_data_path

        target = tmp_path / "custom.db"
        assert resolve_data_path(target) == target


class TestLanguagePairTrie:
    def test_exact_lookup_returns_only_that_pair(self):
        trie = LanguagePairTrie()
        trie.insert("java>python", "a")
        trie.insert("java>go", "b")
        assert trie.search("java>python") == ["a"]

    def test_prefix_lookup_spans_all_targets_for_a_source(self):
        trie = LanguagePairTrie()
        trie.insert("java>python", "a")
        trie.insert("java>go", "b")
        trie.insert("rust>go", "c")
        assert sorted(trie.search_prefix("java>")) == ["a", "b"]

    def test_missing_key_returns_empty_not_an_error(self):
        trie = LanguagePairTrie()
        trie.insert("java>python", "a")
        assert trie.search("cobol>rust") == []
        assert trie.search_prefix("cobol") == []

    def test_duplicate_insert_is_idempotent(self):
        trie = LanguagePairTrie()
        trie.insert("java>python", "a")
        trie.insert("java>python", "a")
        assert trie.search("java>python") == ["a"]
        assert len(trie) == 1

    def test_pair_key_is_case_insensitive(self):
        assert pair_key("Java", "PYTHON") == pair_key("java", "python")


class TestCosineSimilarity:
    def test_identical_code_scores_one(self):
        assert cosine_similarity(JAVA_SRC, JAVA_SRC) == pytest.approx(1.0)

    def test_unrelated_code_scores_near_zero(self):
        assert cosine_similarity("alpha beta gamma", "delta epsilon") == 0.0

    def test_shared_identifiers_score_between(self):
        score = cosine_similarity(
            "def parse_config(path): return path",
            "def parse_config(path, mode): return mode",
        )
        assert 0.0 < score < 1.0

    def test_empty_input_is_zero_not_a_crash(self):
        assert cosine_similarity("", JAVA_SRC) == 0.0


class TestMigrationMemoryRecall:
    def test_remember_then_recall_round_trips(self, memory):
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
            plan="collections",
            score=0.9,
        )
        hits = memory.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert len(hits) == 1
        assert hits[0]["migrated_code"] == PY_SRC
        assert hits[0]["origin"] == "sqlite"

    def test_identical_source_is_an_exact_hit(self, memory):
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
        )
        hits = memory.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert hits[0]["similarity"] == 1.0

    def test_recall_never_crosses_language_pairs(self, memory):
        # The trie filter is a correctness boundary, not an optimisation: a
        # Python migration must not be grounded in a Go precedent.
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="go",
            migrated_code="package main",
        )
        assert (
            memory.recall(
                source_code=JAVA_SRC, source_language="java", target_language="python"
            )
            == []
        )

    def test_dissimilar_code_is_filtered_out(self, tmp_path):
        store = MemoryStore(tmp_path / "s.db")
        mem = MigrationMemory(store, min_similarity=0.9).initialize()
        mem.remember(
            source_code="completely unrelated tokens here",
            source_language="java",
            target_language="python",
            migrated_code="x",
        )
        assert (
            mem.recall(
                source_code=JAVA_SRC, source_language="java", target_language="python"
            )
            == []
        )
        store.close()

    def test_more_similar_precedent_outranks_a_better_scored_one(self, memory):
        # Quality only breaks ties; it must not surface an unrelated migration
        # ahead of the one that actually matches.
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code="RELEVANT",
            score=0.1,
        )
        memory.remember(
            source_code="int x = compute(alpha, beta);",
            source_language="java",
            target_language="python",
            migrated_code="IRRELEVANT",
            score=1.0,
        )
        hits = memory.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert hits[0]["migrated_code"] == "RELEVANT"

    def test_min_score_excludes_low_quality_precedents(self, memory):
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
            score=0.2,
        )
        assert (
            memory.recall(
                source_code=JAVA_SRC,
                source_language="java",
                target_language="python",
                min_score=0.5,
            )
            == []
        )

    def test_recall_survives_a_restart(self, tmp_path):
        # The whole point of Dimension 5: a new process recalls what the old one
        # learned. The trie is derived state, so this also covers its rebuild.
        db = tmp_path / "cross.db"
        first = build_memory(str(db))
        first.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
            score=0.9,
        )
        first.store.close()

        second = build_memory(str(db))
        assert len(second.trie) == 1
        hits = second.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert hits[0]["migrated_code"] == PY_SRC
        second.store.close()

    def test_failures_are_reachable_as_warnings(self, memory):
        memory.record_failure(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            reason="boom",
        )
        assert memory.warnings_for("java", "python")[0]["reason"] == "boom"
        # ...but never surface as groundable prior art.
        assert (
            memory.recall(
                source_code=JAVA_SRC, source_language="java", target_language="python"
            )
            == []
        )

    def test_user_feedback_reaches_recall(self, memory):
        entry_id = memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
        )
        memory.record_feedback(entry_id, "always use pathlib")
        hits = memory.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert hits[0]["user_feedback"] == "always use pathlib"

    def test_build_memory_returns_none_on_an_unusable_path(self, tmp_path):
        # Memory is an optimisation; a bad path must not take migrations down.
        blocker = tmp_path / "file.txt"
        blocker.write_text("not a directory")
        assert build_memory(str(blocker / "nested" / "x.db")) is None


class _StubSemantic:
    """Minimal stand-in for SemanticMigrationMemory.search()."""

    def __init__(self, docs):
        self._docs = docs

    def search(self, query, k=3, where=None):
        return self._docs


class _Doc:
    def __init__(self, metadata):
        self.metadata = metadata
        self.page_content = ""


class TestSemanticMerge:
    def test_semantic_hits_are_merged_and_deduplicated(self, tmp_path):
        store = MemoryStore(tmp_path / "s.db")
        mem = MigrationMemory(store, min_similarity=0.1).initialize()
        entry_id = mem.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
        )
        # One doc duplicates the SQL hit by entry id, one is new.
        mem.attach_semantic(
            _StubSemantic(
                [
                    (_Doc({"entry_id": entry_id, "language": "python"}), 0.99),
                    (_Doc({"entry_id": "other", "language": "python"}), 0.95),
                ]
            )
        )
        hits = mem.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python", k=5
        )
        ids = [h["entry_id"] for h in hits]
        assert ids.count(entry_id) == 1
        assert "other" in ids
        store.close()

    def test_a_broken_semantic_leg_degrades_to_sql_only(self, tmp_path):
        class Broken:
            def search(self, *a, **kw):
                raise RuntimeError("chroma down")

        store = MemoryStore(tmp_path / "s.db")
        mem = MigrationMemory(store, min_similarity=0.1).initialize()
        mem.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
        )
        mem.attach_semantic(Broken())
        hits = mem.recall(
            source_code=JAVA_SRC, source_language="java", target_language="python"
        )
        assert len(hits) == 1 and hits[0]["origin"] == "sqlite"
        store.close()


class TestBloomFilter:
    def test_never_reports_a_false_negative(self):
        # The one-sided guarantee the dedup logic depends on.
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        items = [f"pattern-{i}" for i in range(500)]
        for item in items:
            bloom.add(item)
        assert all(item in bloom for item in items)

    def test_false_positive_rate_stays_near_the_target(self):
        bloom = BloomFilter(capacity=1000, error_rate=0.01)
        for i in range(1000):
            bloom.add(f"in-{i}")
        positives = sum(1 for i in range(5000) if f"out-{i}" in bloom)
        assert positives / 5000 < 0.05  # generous headroom over the 1% target

    def test_survives_serialization(self):
        bloom = BloomFilter(capacity=100, error_rate=0.01)
        bloom.add("alpha")
        restored = BloomFilter.from_bytes(bloom.to_bytes())
        assert "alpha" in restored
        assert restored.num_hashes == bloom.num_hashes
        assert restored.count == 1

    def test_rejects_nonsense_parameters(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=0)
        with pytest.raises(ValueError):
            BloomFilter(capacity=10, error_rate=1.5)


class TestPatternStore:
    @pytest.fixture
    def patterns(self, tmp_path):
        store = MemoryStore(tmp_path / "p.db")
        yield PatternStore(store).initialize()
        store.close()

    def test_first_write_is_new_and_second_is_a_duplicate(self, patterns):
        kwargs = dict(
            source_language="java",
            target_language="python",
            source_snippet="new ArrayList<>()",
            target_snippet="[]",
        )
        assert patterns.add_pattern(**kwargs) is True
        assert patterns.add_pattern(**kwargs) is False
        assert patterns._store.count("patterns") == 1

    def test_duplicates_increment_the_occurrence_count(self, patterns):
        kwargs = dict(
            source_language="java",
            target_language="python",
            source_snippet="HashMap<>",
            target_snippet="{}",
        )
        patterns.add_pattern(**kwargs)
        patterns.add_pattern(**kwargs)
        assert patterns.get_patterns("java", "python")[0]["occurrences"] == 2

    def test_reformatting_is_not_a_new_pattern(self, patterns):
        assert patterns.add_pattern(
            source_language="java",
            target_language="python",
            source_snippet="new  ArrayList<>()",
            target_snippet="[]",
        )
        assert not patterns.add_pattern(
            source_language="java",
            target_language="python",
            source_snippet="new ArrayList<>()",
            target_snippet="[]",
        )

    def test_patterns_are_scoped_per_language_pair(self, patterns):
        for target in ("python", "go"):
            patterns.add_pattern(
                source_language="java",
                target_language=target,
                source_snippet="List<String>",
                target_snippet="x",
            )
        assert len(patterns.get_patterns("java", "python")) == 1
        assert len(patterns.get_patterns("java", "go")) == 1

    def test_bloom_filter_survives_a_restart(self, tmp_path):
        db = tmp_path / "bloom.db"
        store = MemoryStore(db)
        first = PatternStore(store).initialize()
        first.add_pattern(
            source_language="java",
            target_language="python",
            source_snippet="a",
            target_snippet="b",
        )
        store.close()

        store2 = MemoryStore(db)
        second = PatternStore(store2).initialize()
        assert second.seen("java", "python", "a", "b")
        assert (
            second.add_pattern(
                source_language="java",
                target_language="python",
                source_snippet="a",
                target_snippet="b",
            )
            is False
        )
        store2.close()


class TestCheckpointer:
    def test_falls_back_to_memory_saver_without_an_event_loop(self, tmp_path):
        # AsyncSqliteSaver binds to a running loop at construction; synchronous
        # callers (tests, scripts) must still get a working checkpointer.
        class _S:
            memory_enabled = True
            checkpointer_enabled = True
            checkpoint_db_path = str(tmp_path / "cp.db")

        assert type(build_checkpointer(_S())).__name__ in {
            "MemorySaver",
            "InMemorySaver",
        }

    def test_disabled_returns_none(self, tmp_path):
        class _S:
            memory_enabled = True
            checkpointer_enabled = False
            checkpoint_db_path = str(tmp_path / "cp.db")

        assert build_checkpointer(_S()) is None

    @pytest.mark.asyncio
    async def test_uses_the_async_sqlite_saver_inside_a_loop(self, tmp_path):
        class _S:
            memory_enabled = True
            checkpointer_enabled = True
            checkpoint_db_path = str(tmp_path / "cp.db")

        saver = build_checkpointer(_S())
        assert type(saver).__name__ == "AsyncSqliteSaver"
        await saver.conn.close()

    def test_graph_without_a_checkpointer_needs_no_thread_id(self):
        # Default must stay invocable without config — attaching a checkpointer
        # by default would break every existing direct caller.
        app = build_migration_graph()
        assert app.checkpointer is None

    @pytest.mark.asyncio
    async def test_checkpointed_graph_persists_state_across_instances(self, tmp_path):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.graph import END, StateGraph
        from typing_extensions import TypedDict

        import aiosqlite

        class _S(TypedDict, total=False):
            n: int

        async def bump(state):
            return {"n": state.get("n", 0) + 1}

        def compile_with(saver):
            flow = StateGraph(_S)
            flow.add_node("bump", bump)
            flow.set_entry_point("bump")
            flow.add_edge("bump", END)
            return flow.compile(checkpointer=saver)

        db = str(tmp_path / "graph.db")
        config = {"configurable": {"thread_id": "session-1"}}

        first = AsyncSqliteSaver(aiosqlite.connect(db, check_same_thread=False))
        await compile_with(first).ainvoke({"n": 0}, config=config)
        await first.conn.close()

        second = AsyncSqliteSaver(aiosqlite.connect(db, check_same_thread=False))
        restored = await compile_with(second).aget_state(config)
        assert restored.values["n"] == 1
        await second.conn.close()


class TestPipelineIntegration:
    def _pipeline(self, memory, **overrides):
        from config import get_settings

        settings = get_settings().model_copy(update={"memory_enabled": True, **overrides})
        return Pipeline(None, None, settings=settings, memory=memory)

    def _state(self, **overrides):
        base = dict(
            source_code=JAVA_SRC,
            source_language="java",
            source_version="8",
            target_language="python",
            target_version="3.12",
        )
        base.update(overrides)
        return MigrationState(**base)

    @pytest.mark.asyncio
    async def test_recall_populates_hits_and_grounds_the_prompt(self, memory):
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
            score=0.9,
        )
        pipeline = self._pipeline(memory)
        state = self._state()
        await pipeline._recall(state)

        assert len(state.memory_hits) == 1
        assert "Similar past migrations" in state.rag_context
        assert PY_SRC[:20] in state.rag_context

    @pytest.mark.asyncio
    async def test_recall_appends_after_existing_retrieved_docs(self, memory):
        # Official documentation must keep precedence over "what we did before".
        memory.remember(
            source_code=JAVA_SRC,
            source_language="java",
            target_language="python",
            migrated_code=PY_SRC,
        )
        pipeline = self._pipeline(memory)
        state = self._state(rag_context="## Official docs\nuse list")
        await pipeline._recall(state)
        assert state.rag_context.index("Official docs") < state.rag_context.index(
            "Similar past migrations"
        )

    @pytest.mark.asyncio
    async def test_recall_is_a_noop_without_memory(self):
        pipeline = self._pipeline(None)
        state = self._state()
        await pipeline._recall(state)
        assert state.memory_hits == []
        assert state.rag_context == ""

    @pytest.mark.asyncio
    async def test_a_broken_store_does_not_fail_the_run(self):
        class Broken:
            def recall(self, **kwargs):
                raise RuntimeError("db locked")

        pipeline = self._pipeline(Broken())
        state = self._state()
        await pipeline._recall(state)  # must not raise
        assert state.memory_hits == []

    @pytest.mark.asyncio
    async def test_successful_run_is_persisted(self, memory):
        pipeline = self._pipeline(memory)
        state = self._state(
            migrated_code=PY_SRC,
            validation_result={"valid": True},
            session_id="s1",
        )
        await pipeline._persist(state)
        assert memory.store.count("migrations") == 1
        assert memory.store.count("failures") == 0

    @pytest.mark.asyncio
    async def test_failed_run_is_recorded_as_a_failure_not_a_precedent(self, memory):
        pipeline = self._pipeline(memory)
        state = self._state(migrated_code="x", validation_result={"valid": False})
        await pipeline._persist(state)
        assert memory.store.count("migrations") == 0
        assert memory.store.count("failures") == 1

    @pytest.mark.asyncio
    async def test_run_with_errors_is_not_stored_as_prior_art(self, memory):
        pipeline = self._pipeline(memory)
        state = self._state(migrated_code=PY_SRC)
        state.record_error("MigratorAgent", "exploded")
        await pipeline._persist(state)
        assert memory.store.count("migrations") == 0

    @pytest.mark.asyncio
    async def test_recall_then_persist_closes_the_loop(self, tmp_path):
        # End to end: run 1 stores, run 2 (fresh memory object, same file)
        # recalls it. This is the behaviour Dimension 5 exists to provide.
        db = tmp_path / "loop.db"
        first = build_memory(str(db))
        pipeline = self._pipeline(first)
        await pipeline._persist(
            self._state(migrated_code=PY_SRC, validation_result={"valid": True})
        )
        first.store.close()

        second = build_memory(str(db))
        state = self._state()
        await self._pipeline(second)._recall(state)
        assert state.memory_hits[0]["migrated_code"] == PY_SRC
        second.store.close()


class _E2ELLM:
    """Stub covering every agent prompt on the default sequential path."""

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None) -> str:
        import json

        if prompt.startswith("Analyze this"):
            return json.dumps(
                {
                    "deprecated_patterns": [],
                    "migration_challenges": [],
                    "key_constructs": ["Foo"],
                    "summary": "Stub analysis.",
                }
            )
        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps({"plan_summary": "plan", "steps": [], "risk_areas": []})
        return PY_SRC


class TestEndToEndPersistence:
    """The full pipeline against a real checkpointed graph — no services."""

    def _settings(self, tmp_path, **overrides):
        from config import get_settings

        return get_settings().model_copy(
            update={
                "memory_enabled": True,
                "checkpointer_enabled": True,
                "memory_db_path": str(tmp_path / "mem.db"),
                "checkpoint_db_path": str(tmp_path / "cp.db"),
                "cache_enabled": False,
                "enable_validation": False,
                "enable_rag": False,
                "enable_reflection": False,
                "parallel_enabled": False,
                "enable_streaming": False,
                **overrides,
            }
        )

    def _state(self):
        return MigrationState(
            source_code=JAVA_SRC,
            source_language="java",
            source_version="8",
            target_language="python",
            target_version="3.12",
        )

    @pytest.mark.asyncio
    async def test_run_completes_and_learns_across_sessions(self, tmp_path):
        from graph import nodes as graph_nodes

        settings = self._settings(tmp_path)

        # --- session 1: cold. Nothing to recall; the outcome is recorded. ---
        first_memory = build_memory(settings.memory_db_path)
        pipeline = Pipeline(
            _E2ELLM(), None, settings=settings, memory=first_memory
        )
        try:
            result = await pipeline.run(self._state())
        finally:
            await pipeline.aclose()
            graph_nodes.set_llm_client(None)

        # The thread_id contract held all the way through astream.
        assert result.migrated_code
        assert result.session_id
        assert result.memory_hits == []
        assert first_memory.store.count("migrations") == 1
        first_memory.store.close()

        # The checkpointer actually wrote to its own database.
        assert (tmp_path / "cp.db").exists()
        assert (tmp_path / "cp.db").stat().st_size > 0

        # --- session 2: a brand new process-equivalent recalls session 1. ---
        second_memory = build_memory(settings.memory_db_path)
        pipeline2 = Pipeline(
            _E2ELLM(), None, settings=settings, memory=second_memory
        )
        try:
            result2 = await pipeline2.run(self._state())
        finally:
            await pipeline2.aclose()
            graph_nodes.set_llm_client(None)

        assert result2.memory_hits, "session 2 should recall session 1"
        assert result2.memory_hits[0]["similarity"] == 1.0
        # Re-running the same source overwrites its record rather than piling up.
        assert second_memory.store.count("migrations") == 1
        second_memory.store.close()

    @pytest.mark.asyncio
    async def test_run_without_memory_still_works(self, tmp_path):
        from graph import nodes as graph_nodes

        pipeline = Pipeline(
            _E2ELLM(),
            None,
            settings=self._settings(tmp_path, memory_enabled=False),
            memory=None,
        )
        try:
            result = await pipeline.run(self._state())
        finally:
            await pipeline.aclose()
            graph_nodes.set_llm_client(None)
        assert result.migrated_code
        assert result.memory_hits == []

    @pytest.mark.asyncio
    async def test_caller_supplied_session_id_is_preserved(self, tmp_path):
        from graph import nodes as graph_nodes

        memory = build_memory(self._settings(tmp_path).memory_db_path)
        pipeline = Pipeline(
            _E2ELLM(), None, settings=self._settings(tmp_path), memory=memory
        )
        state = self._state()
        state.session_id = "caller-owned-id"
        try:
            result = await pipeline.run(state)
        finally:
            await pipeline.aclose()
            graph_nodes.set_llm_client(None)
        assert result.session_id == "caller-owned-id"
        memory.store.close()


class TestQualityScore:
    def test_validated_code_outscores_unvalidated(self):
        valid = MigrationState(
            source_code="a",
            source_language="java",
            source_version="8",
            target_language="python",
            target_version="3.12",
            validation_result={"valid": True},
        )
        invalid = valid.model_copy(update={"validation_result": {"valid": False}})
        assert _quality_score(valid) > _quality_score(invalid)

    def test_retries_reduce_the_score(self):
        clean = MigrationState(
            source_code="a",
            source_language="java",
            source_version="8",
            target_language="python",
            target_version="3.12",
            validation_result={"valid": True},
        )
        retried = clean.model_copy(deep=True)
        retried.record_success("FixerAgent", "fixed")
        assert _quality_score(retried) < _quality_score(clean)

    def test_score_stays_within_bounds(self):
        state = MigrationState(
            source_code="a",
            source_language="java",
            source_version="8",
            target_language="python",
            target_version="3.12",
            validation_result={"valid": False},
        )
        for _ in range(20):
            state.record_success("FixerAgent", "fixed")
        assert 0.0 <= _quality_score(state) <= 1.0


class TestSummarizeHits:
    def test_renders_feedback_and_code(self):
        rendered = summarize_hits(
            [
                {
                    "source_language": "java",
                    "target_language": "python",
                    "target_version": "3.12",
                    "similarity": 0.9,
                    "plan": "convert collections",
                    "user_feedback": "prefer pathlib",
                    "migrated_code": PY_SRC,
                }
            ]
        )
        assert "prefer pathlib" in rendered
        assert "convert collections" in rendered
        assert "java -> python" in rendered

    def test_empty_hits_render_empty(self):
        assert summarize_hits([]) == ""


class TestThreadSafety:
    @pytest.mark.asyncio
    async def test_concurrent_writes_do_not_corrupt_the_store(self, memory):
        # The parallel fan-out reaches this store from worker threads.
        async def write(index):
            await asyncio.to_thread(
                memory.remember,
                source_code=f"code {index}",
                source_language="java",
                target_language="python",
                migrated_code=f"out {index}",
            )

        await asyncio.gather(*(write(i) for i in range(25)))
        assert memory.store.count("migrations") == 25
