"""Tests for cross-session migration memory and the semantic_search tool."""

import pytest

from agents.tools.semantic_search import SemanticSearchTool
from models.state import MigrationState
from rag.migration_memory import MigrationMemory
from runtime.agent_observer import ObserverAgent


def _state(**overrides) -> MigrationState:
    base = dict(
        source_code="print 'hi'",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
        migrated_code="print('hi')",
        inline_plan="Use the print function.",
        validation_result={"valid": True, "errors": [], "warnings": []},
    )
    base.update(overrides)
    return MigrationState(**base)


class _FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


class _FakeMemory:
    """In-memory stand-in for the Chroma-backed store."""

    def __init__(self, hits=None):
        self.hits = hits or []
        self.remembered: list[dict] = []
        self.searches: list[tuple] = []

    def remember(self, **kwargs):
        self.remembered.append(kwargs)
        return "mem-1"

    def search(self, query, k=3, where=None):
        self.searches.append((query, k, where))
        return self.hits


class TestMigrationMemoryEntryId:
    def test_same_migration_yields_the_same_id(self):
        # Stable ids make re-running a migration overwrite its memory rather than
        # accumulate near-duplicate entries that all match the next query.
        first = MigrationMemory.entry_id("code", "java", "21")
        second = MigrationMemory.entry_id("code", "java", "21")
        assert first == second

    def test_different_target_yields_a_different_id(self):
        assert MigrationMemory.entry_id("code", "java", "21") != MigrationMemory.entry_id(
            "code", "java", "17"
        )


class TestSemanticSearchTool:
    @pytest.mark.asyncio
    async def test_returns_precedents_above_the_threshold(self, settings_override):
        settings_override(memory_min_score=0.8)
        memory = _FakeMemory(
            hits=[
                (
                    _FakeDoc(
                        "old",
                        {
                            "language": "python",
                            "version": "3.12",
                            "plan_summary": "print fn",
                            "migrated_code": "print('hi')",
                        },
                    ),
                    0.95,
                )
            ]
        )
        result = await SemanticSearchTool(memory)(query="print statement")
        assert result.success
        assert result.data["precedents"][0]["migrated_code"] == "print('hi')"

    @pytest.mark.asyncio
    async def test_weak_matches_are_dropped(self, settings_override):
        # A loosely-similar past migration is worse than none: it would ground new
        # code in a precedent that doesn't actually apply.
        settings_override(memory_min_score=0.8)
        memory = _FakeMemory(hits=[(_FakeDoc("unrelated", {}), 0.4)])
        result = await SemanticSearchTool(memory)(query="q")
        assert result.success
        assert result.data["precedents"] == []

    @pytest.mark.asyncio
    async def test_missing_memory_fails_cleanly(self):
        result = await SemanticSearchTool(None)(query="q")
        assert not result.success

    @pytest.mark.asyncio
    async def test_empty_query_is_rejected(self):
        result = await SemanticSearchTool(_FakeMemory())(query="  ")
        assert not result.success

    def test_filter_uses_and_for_multiple_keys(self):
        # Chroma rejects a multi-key filter without an explicit $and.
        assert SemanticSearchTool._build_filter("java", "21") == {
            "$and": [{"language": "java"}, {"version": "21"}]
        }

    def test_filter_is_flat_for_one_key(self):
        assert SemanticSearchTool._build_filter("java", "") == {"language": "java"}

    def test_filter_is_none_when_unconstrained(self):
        assert SemanticSearchTool._build_filter("", "") is None


class TestObserverRemembers:
    @pytest.mark.asyncio
    async def test_clean_migration_is_remembered(self, settings_override):
        settings_override(enable_migration_memory=True)
        memory = _FakeMemory()
        agent = ObserverAgent(None, {"migration_memory": memory})
        result = await agent.run(_state())

        assert result.details["remembered"] is True
        assert memory.remembered[0]["migrated_code"] == "print('hi')"
        assert memory.remembered[0]["plan_summary"] == "Use the print function."

    @pytest.mark.asyncio
    async def test_invalid_code_is_not_remembered(self, settings_override):
        # semantic_search presents hits as known-good prior art, so storing a
        # failed migration would launder a failure into future grounding.
        settings_override(enable_migration_memory=True)
        memory = _FakeMemory()
        agent = ObserverAgent(None, {"migration_memory": memory})
        state = _state(validation_result={"valid": False, "errors": [{}], "warnings": []})
        result = await agent.run(state)

        assert result.details["remembered"] is False
        assert memory.remembered == []

    @pytest.mark.asyncio
    async def test_run_with_errors_is_not_remembered(self, settings_override):
        settings_override(enable_migration_memory=True)
        memory = _FakeMemory()
        agent = ObserverAgent(None, {"migration_memory": memory})
        state = _state()
        state.record_error("MigratorAgent", "boom")
        result = await agent.run(state)

        assert result.details["remembered"] is False
        assert memory.remembered == []

    @pytest.mark.asyncio
    async def test_nothing_remembered_without_migrated_code(self, settings_override):
        settings_override(enable_migration_memory=True)
        memory = _FakeMemory()
        agent = ObserverAgent(None, {"migration_memory": memory})
        result = await agent.run(_state(migrated_code=""))
        assert result.details["remembered"] is False

    @pytest.mark.asyncio
    async def test_disabled_setting_skips_memory_entirely(self, settings_override):
        settings_override(enable_migration_memory=False)
        memory = _FakeMemory()
        agent = ObserverAgent(None, {"migration_memory": memory})
        result = await agent.run(_state())
        assert result.details["remembered"] is False
        assert memory.remembered == []

    @pytest.mark.asyncio
    async def test_observer_survives_a_broken_memory_store(self, settings_override):
        # Memory is an optimization; a store that is down must not fail the run.
        settings_override(enable_migration_memory=True)

        class _BrokenMemory:
            def remember(self, **kwargs):
                raise RuntimeError("chroma is down")

        agent = ObserverAgent(None, {"migration_memory": _BrokenMemory()})
        result = await agent.run(_state())
        assert result.success
        assert result.details["remembered"] is False

    @pytest.mark.asyncio
    async def test_observer_still_reports_metrics_without_memory(self):
        agent = ObserverAgent(None)
        state = _state()
        state.record_success("AnalyzerAgent", "ok", duration_ms=5)
        result = await agent.run(state)
        assert result.success
        assert result.details["agents_run"] == 1
