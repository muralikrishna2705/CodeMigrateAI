"""The delta contract: what a node returns, and how the reducers fold it.

Two halves of one invariant. Nodes return only what they changed, and
``GraphState``'s reducers say how each change accumulates. Either half alone is
a bug — reducers with full-state returns duplicate history, delta returns
without reducers make concurrent branches conflict — so these tests pin both
and the relationship between them.
"""

import asyncio
import operator

import pytest
from graph.nodes import hydrate_state, state_delta, writeback
from graph.state import (
    ACCUMULATING_FIELDS,
    COUNTER_FIELDS,
    GraphState,
    _keep_highest,
    _merge_dicts,
)
from models.state import AgentReport, MigrationState


def _state(**overrides) -> dict:
    base = {
        "source_code": "print('x')",
        "source_language": "python",
        "source_version": "2.7",
        "target_language": "python",
        "target_version": "3.12",
        "migration_type": "upgrade_version",
        "reports": [],
        "errors": [],
        "agents_completed": [],
        "retrieval_requests": [],
        "reretrieval_count": 0,
        "rag_context": "",
        "inline_plan": "",
        "migrated_code": "",
    }
    base.update(overrides)
    return base


class TestReducerDeclarations:
    """The annotations are the contract; assert they say what we think."""

    def test_run_history_accumulates(self):
        for field in ACCUMULATING_FIELDS:
            metadata = GraphState.__annotations__[field].__metadata__
            assert operator.add in metadata, f"{field} lost its additive reducer"

    def test_loop_counters_take_the_highest(self):
        for field in COUNTER_FIELDS:
            metadata = GraphState.__annotations__[field].__metadata__
            assert _keep_highest in metadata, f"{field} lost its max reducer"

    def test_retrieval_requests_is_deliberately_not_additive(self):
        # retrieve_node drains the queue by writing []. Under operator.add an
        # empty list is a no-op, so the queue would never clear and the
        # re-retrieval loop would run to its budget on every migration.
        assert not hasattr(GraphState.__annotations__["retrieval_requests"], "__metadata__")

    def test_migration_inputs_carry_no_reducer(self):
        # Nothing writes them, so nothing has to fold them.
        for field in ("source_code", "source_language", "target_version"):
            assert not hasattr(GraphState.__annotations__[field], "__metadata__")


class TestReducerBehaviour:
    def test_counters_are_idempotent(self):
        # Writers emit the absolute value, so replaying a delta must not
        # advance the budget a second time.
        assert _keep_highest(2, 2) == 2

    def test_a_consumed_budget_survives_a_sibling_that_did_not_consume_it(self):
        assert _keep_highest(1, 0) == 1
        assert _keep_highest(0, 1) == 1

    def test_counters_tolerate_an_unset_channel(self):
        assert _keep_highest(None, 1) == 1
        assert _keep_highest(None, None) == 0

    def test_metrics_merge_key_wise(self):
        # AnalyzerAgent writes the base metrics; DeepAnalyzerAgent adds its own
        # key. Last-write-wins would drop one of them.
        merged = _merge_dicts({"complexity": "high"}, {"deep_analysis": {"a": 1}})
        assert merged == {"complexity": "high", "deep_analysis": {"a": 1}}

    def test_never_analyzed_stays_none(self):
        # Downstream distinguishes "no metrics" from "empty metrics".
        assert _merge_dicts(None, None) is None


class TestWriteback:
    """What a node hands back to the channels."""

    def _run(self, state, mutate):
        mig_state = hydrate_state(state)
        mutate(mig_state)
        return writeback(state, mig_state)

    def test_unchanged_fields_are_absent(self):
        delta = self._run(_state(), lambda s: None)
        assert delta == {}

    def test_inputs_are_never_echoed_back(self):
        # The clearest single signal that the contract holds: an agent that
        # produced a report says nothing about the source code it read.
        delta = self._run(
            _state(),
            lambda s: s.reports.append(AgentReport(agent="A", status="success", summary="done")),
        )
        for field in ("source_code", "source_language", "target_version"):
            assert field not in delta

    def test_only_the_new_reports_are_emitted(self):
        state = _state(
            reports=[{"agent": "First", "status": "success", "summary": "done"}]
        )
        delta = self._run(
            state,
            lambda s: s.reports.append(AgentReport(agent="Second", status="success", summary="done")),
        )
        # Not both — the reducer appends whatever it is handed, so returning the
        # full list would leave two copies of First in the channel.
        assert [r["agent"] for r in delta["reports"]] == ["Second"]

    def test_the_inbound_state_is_not_mutated(self):
        state = _state()
        snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in state.items()}
        self._run(
            state, lambda s: s.reports.append(AgentReport(agent="A", status="success", summary="done"))
        )
        # The dict belongs to the channels. Writing through to it bypasses the
        # reducers entirely, which is invisible until a branch runs concurrently.
        assert state == snapshot

    def test_a_changed_scalar_is_emitted(self):
        delta = self._run(_state(), lambda s: setattr(s, "migrated_code", "x = 1"))
        assert delta == {"migrated_code": "x = 1"}

    def test_a_scalar_reset_to_empty_is_still_a_change(self):
        # Clearing rag_context is a real edit, not an absence of one.
        delta = self._run(
            _state(rag_context="old"), lambda s: setattr(s, "rag_context", "")
        )
        assert delta == {"rag_context": ""}


class TestStateDelta:
    """The bridge used while the fan-out still merges whole states."""

    def test_accumulating_fields_yield_only_their_tail(self):
        base = _state(reports=[{"agent": "A"}])
        merged = _state(reports=[{"agent": "A"}, {"agent": "B"}])
        assert state_delta(base, merged)["reports"] == [{"agent": "B"}]

    def test_unchanged_keys_are_dropped(self):
        base = _state(rag_context="same")
        assert "rag_context" not in state_delta(base, _state(rag_context="same"))

    def test_a_shorter_list_yields_no_tail(self):
        base = _state(errors=["a", "b"])
        assert "errors" not in state_delta(base, _state(errors=["a"]))


class TestNodeSkipPaths:
    """Every early return in the node layer is a delta, not a state."""

    @pytest.mark.asyncio
    async def test_reflection_gate_contributes_nothing(self):
        from graph.nodes import reflect_node

        assert await reflect_node(_state(enable_reflection=False)) == {}

    @pytest.mark.asyncio
    async def test_service_validation_gate_contributes_nothing(self):
        from graph.nodes import service_validate_node

        assert await service_validate_node(_state(enable_validation=False)) == {}
        assert (
            await service_validate_node(
                _state(enable_validation=True, migrated_code="x", errors=["boom"])
            )
            == {}
        )

    @pytest.mark.asyncio
    async def test_an_open_circuit_contributes_nothing(self):
        from runtime import agent_recovery
        from runtime.agent_providers import Provider
        from runtime.agent_runtime import Runtime

        for _ in range(10):
            agent_recovery.record_failure("AnalyzerAgent")
        assert agent_recovery.is_circuit_open("AnalyzerAgent")
        try:
            assert await Runtime(Provider()).execute("AnalyzerAgent", _state()) == {}
        finally:
            agent_recovery.reset()

    @pytest.mark.asyncio
    async def test_an_unknown_agent_reports_only_its_error(self):
        from runtime.agent_providers import Provider
        from runtime.agent_runtime import Runtime

        state = _state(reports=[{"agent": "Earlier", "status": "success"}])
        delta = await Runtime(Provider()).execute("NoSuchAgent", state)
        assert delta == {"errors": ["Agent NoSuchAgent not found"]}
        # Specifically not the accumulated reports: an additive reducer would
        # duplicate every one of them.
        assert "reports" not in delta


class TestCounterLoopsTerminate:
    """The wrapper nodes advance their budgets from the *inbound* state.

    Reading a counter back off the delta returns 0 every time — the agent never
    touches it — so the loop would never reach its ceiling. The failure mode is
    a hang, not an exception, which is why it gets its own tests.
    """

    @pytest.mark.asyncio
    async def test_fix_advances_the_retry_budget(self, monkeypatch):
        from graph import nodes

        async def _fixer(state):
            return {}

        monkeypatch.setattr(nodes, "_fixer_node", _fixer)
        delta = await nodes.fix_node(_state(retry_count=1, migrated_code="old"))
        assert delta["retry_count"] == 2
        assert delta["best_effort_code"] == "old"

    @pytest.mark.asyncio
    async def test_re_retrieval_advances_and_drains_the_queue(self, monkeypatch):
        from graph import nodes

        async def _retriever(state):
            return {"rag_context": "docs"}

        monkeypatch.setattr(nodes, "_retriever_node", _retriever)
        delta = await nodes.retrieve_node(
            _state(retrieval_requests=["asyncio"], reretrieval_count=0)
        )
        assert delta["reretrieval_count"] == 1
        assert delta["retrieval_requests"] == []

    @pytest.mark.asyncio
    async def test_a_first_pass_retrieval_does_not_consume_the_budget(
        self, monkeypatch
    ):
        from graph import nodes

        async def _retriever(state):
            return {"rag_context": "docs"}

        monkeypatch.setattr(nodes, "_retriever_node", _retriever)
        delta = await nodes.retrieve_node(_state(retrieval_requests=[]))
        assert "reretrieval_count" not in delta

    @pytest.mark.asyncio
    async def test_reflection_regeneration_advances_and_clears_the_verdict(
        self, monkeypatch
    ):
        from graph import nodes

        async def _migrator(state):
            return {"migrated_code": "new"}

        monkeypatch.setattr(nodes, "_migrate_node", _migrator)
        delta = await nodes.migrate_node(
            _state(
                reflection_feedback="too many stubs",
                reflection_recommendation="revise",
                reflection_count=0,
            )
        )
        assert delta["reflection_count"] == 1
        assert delta["reflection_feedback"] == ""
        assert delta["reflection_recommendation"] == "pass"
