"""Tests for dynamic orchestration: decomposition, fan-out, and fan-in merge.

Everything runs offline against scripted stub LLMs — no Ollama/validator service —
including an end-to-end fan-out through the real compiled graph and subgraphs.
"""

import asyncio
import json

import pytest

from agents.orchestrator_agent import OrchestratorAgent
from graph.conditions import orchestrate_condition
from graph.subgraphs import SUBGRAPH_TASKS, resolve_tasks
from langgraph.types import Send
from models.state import MigrationState


def _state(**overrides) -> MigrationState:
    base = dict(
        source_code="x = 1",
        source_language="python",
        source_version="3.8",
        target_language="python",
        target_version="3.12",
    )
    base.update(overrides)
    return MigrationState(**base)


def _graph_state(**overrides) -> dict:
    base = {
        "source_code": "x = 1",
        "source_language": "python",
        "source_version": "3.8",
        "target_language": "python",
        "target_version": "3.12",
        "reports": [],
        "errors": [],
        "agents_completed": [],
        "retrieval_requests": [],
    }
    base.update(overrides)
    return base


class _StubSubgraph:
    """A compiled-graph stand-in.

    Returns a *new* accumulated state rather than mutating the one it was
    handed, because that is what a real compiled subgraph does: its nodes
    return deltas and LangGraph folds them into fresh channel values. A stub
    that mutated in place would make the caller's before/after diff empty and
    quietly assert nothing.
    """

    def __init__(self, name, mutate=None, delay=0.0, raises=None):
        self.name = name
        self.mutate = mutate
        self.delay = delay
        self.raises = raises
        self.seen_states = []

    async def ainvoke(self, state):
        self.seen_states.append(state)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises:
            raise self.raises
        result = dict(state)
        if self.mutate:
            self.mutate(result)
        return result


class TestOrchestratorAgent:
    @pytest.mark.asyncio
    async def test_deep_analysis_decomposes_into_parallel_tasks(self):
        state = _state(route_plan={"deep_analyze": True})
        result = await OrchestratorAgent(None).run(state)

        assert result.success
        assert state.parallel_tasks == ["analysis", "retrieval"]
        assert state.route_plan["parallelizable"] is True

    @pytest.mark.asyncio
    async def test_no_deep_analysis_stays_sequential(self):
        state = _state(route_plan={"deep_analyze": False})
        await OrchestratorAgent(None).run(state)

        # A single sub-task is not concurrent work, so parallel_tasks stays empty
        # and the run falls through to the original sequential path.
        assert state.parallel_tasks == []
        assert state.route_plan["parallelizable"] is False
        assert state.route_plan["sub_tasks"] == ["retrieval"]

    @pytest.mark.asyncio
    async def test_preserves_dispatcher_route_plan(self):
        state = _state(route_plan={"deep_analyze": True, "complexity": "high"})
        await OrchestratorAgent(None).run(state)

        assert state.route_plan["complexity"] == "high"
        assert state.route_plan["deep_analyze"] is True

    @pytest.mark.asyncio
    async def test_missing_route_plan_defaults_to_sequential(self):
        state = _state()
        await OrchestratorAgent(None).run(state)
        assert state.parallel_tasks == []

    @pytest.mark.asyncio
    async def test_every_planned_task_is_a_known_subgraph(self):
        state = _state(route_plan={"deep_analyze": True})
        await OrchestratorAgent(None).run(state)
        assert all(task in SUBGRAPH_TASKS for task in state.parallel_tasks)


class TestOrchestrateCondition:
    def test_parallel_tasks_become_one_send_each(self):
        state = _graph_state(
            parallel_enabled=True, parallel_tasks=["analysis", "retrieval"]
        )
        sends = orchestrate_condition(state)

        assert all(isinstance(s, Send) for s in sends)
        assert [s.node for s in sends] == ["branch", "branch"]
        # Same target node, different task: which sub-task a branch runs travels
        # in its own Send payload, not in shared state.
        assert [s.arg["branch_task"] for s in sends] == ["analysis", "retrieval"]

    def test_each_branch_receives_the_inbound_state(self):
        state = _graph_state(
            parallel_enabled=True,
            parallel_tasks=["analysis", "retrieval"],
            code_metrics={"complexity": "high"},
        )
        for send in orchestrate_condition(state):
            assert send.arg["source_code"] == "x = 1"
            assert send.arg["code_metrics"] == {"complexity": "high"}

    def test_max_parallel_tasks_caps_the_fan_out(self):
        # Send has no concurrency ceiling of its own — every dispatched branch
        # runs in the same superstep — so capping the number of Sends is what
        # keeps this setting meaning what it always meant.
        state = _graph_state(
            parallel_enabled=True,
            parallel_tasks=["analysis", "retrieval", "reflection"],
            max_parallel_tasks=2,
        )
        assert len(orchestrate_condition(state)) == 2

    def test_a_nonsense_ceiling_still_dispatches_something(self):
        state = _graph_state(
            parallel_enabled=True,
            parallel_tasks=["analysis", "retrieval"],
            max_parallel_tasks=0,
        )
        assert len(orchestrate_condition(state)) == 1

    def test_disabled_falls_back_to_sequential(self):
        state = {
            "parallel_enabled": False,
            "parallel_tasks": ["analysis", "retrieval"],
            "route_plan": {"deep_analyze": True},
        }
        assert orchestrate_condition(state) == "deep_analyze"

    def test_single_task_falls_back_to_sequential(self):
        state = {
            "parallel_enabled": True,
            "parallel_tasks": ["retrieval"],
            "route_plan": {"deep_analyze": False},
        }
        assert orchestrate_condition(state) == "retrieve"

    def test_unseeded_state_takes_sequential_path(self):
        # A direct-graph run never seeds parallel_enabled; it must behave exactly
        # as it did before orchestration existed.
        assert orchestrate_condition({"code_metrics": {"complexity": "high"}}) == (
            "deep_analyze"
        )


class TestSubgraphs:
    def test_every_catalog_entry_compiles(self):
        for name, builder in SUBGRAPH_TASKS.items():
            app = builder()
            assert app is not None, name

    def test_builders_are_cached(self):
        from graph.subgraphs import build_analysis_subgraph

        assert build_analysis_subgraph() is build_analysis_subgraph()

    def test_resolve_tasks_maps_names_to_subgraphs(self):
        resolved = resolve_tasks(["analysis", "retrieval"])
        assert [name for name, _ in resolved] == ["analysis", "retrieval"]

    def test_unknown_task_is_skipped_not_raised(self):
        # Task names can come from an LLM decomposition; one bad name must not
        # abort a migration the remaining tasks can still complete.
        resolved = resolve_tasks(["analysis", "hallucinated"])
        assert [name for name, _ in resolved] == ["analysis"]

    def test_resolve_tasks_handles_empty(self):
        assert resolve_tasks([]) == []
        assert resolve_tasks(None) == []


class TestSubgraphBranchNode:
    """One fanned-out invocation: what it contributes, and how it fails.

    LangGraph aborts the whole superstep on an unhandled exception, so the
    "a failed sub-task contributes nothing and the run continues" guarantee —
    previously enforced by BranchResult swallowing branch errors — now has to
    live inside this node.
    """

    def _patch(self, monkeypatch, **tasks):
        monkeypatch.setattr(
            "graph.subgraphs.SUBGRAPH_TASKS",
            {name: (lambda s=sub: s) for name, sub in tasks.items()},
        )

    @pytest.mark.asyncio
    async def test_contributes_its_subgraph_delta(self, monkeypatch):
        def _mutate(state):
            state["rag_context"] = "docs"
            state["agents_completed"] = state["agents_completed"] + ["RetrieverAgent"]

        self._patch(monkeypatch, retrieval=_StubSubgraph("retrieval", mutate=_mutate))
        from graph.nodes import subgraph_branch_node

        delta = await subgraph_branch_node(
            _graph_state(branch_task="retrieval", agents_completed=["AnalyzerAgent"])
        )

        assert delta["rag_context"] == "docs"
        # Only this branch's agent, not the one that ran before the fan-out —
        # the additive reducer would otherwise record AnalyzerAgent twice.
        assert delta["agents_completed"] == ["RetrieverAgent"]

    @pytest.mark.asyncio
    async def test_records_what_it_ran(self, monkeypatch):
        self._patch(monkeypatch, analysis=_StubSubgraph("analysis"))
        from graph.nodes import subgraph_branch_node

        delta = await subgraph_branch_node(_graph_state(branch_task="analysis"))
        record = delta["subgraph_results"][0]
        assert record["task"] == "analysis"
        assert record["ok"] is True
        assert "duration_ms" in record

    @pytest.mark.asyncio
    async def test_a_failing_branch_degrades_instead_of_raising(self, monkeypatch):
        self._patch(
            monkeypatch,
            analysis=_StubSubgraph("analysis", raises=RuntimeError("boom")),
        )
        from graph.nodes import subgraph_branch_node

        delta = await subgraph_branch_node(_graph_state(branch_task="analysis"))

        record = delta["subgraph_results"][0]
        assert record["ok"] is False
        assert "boom" in record["error"]
        # The failure is visible in the report but contributes no state, so the
        # sibling branch's work and the migration both survive.
        assert "rag_context" not in delta

    @pytest.mark.asyncio
    async def test_an_unknown_task_is_recorded_not_raised(self, monkeypatch):
        self._patch(monkeypatch, analysis=_StubSubgraph("analysis"))
        from graph.nodes import subgraph_branch_node

        delta = await subgraph_branch_node(_graph_state(branch_task="hallucinated"))
        assert delta["subgraph_results"][0]["ok"] is False
        assert delta["subgraph_results"][0]["error"] == "unknown task"

    @pytest.mark.asyncio
    async def test_the_inbound_state_is_left_untouched(self, monkeypatch):
        """What makes it safe for concurrent branches to share one state dict.

        The Sends hand every branch the same object without copying. That is
        sound only while a branch reads it and returns a delta; a branch that
        wrote through would be visible to its siblings mid-flight, which is the
        non-determinism the old implementation spent a deep copy per branch to
        avoid.
        """
        self._patch(
            monkeypatch,
            analysis=_StubSubgraph(
                "analysis", mutate=lambda s: s.update(code_metrics={"deep": True})
            ),
        )
        from graph.nodes import subgraph_branch_node

        state = _graph_state(branch_task="analysis", code_metrics={"complexity": "high"})
        delta = await subgraph_branch_node(state)

        assert delta["code_metrics"] == {"deep": True}
        assert state["code_metrics"] == {"complexity": "high"}
        assert state["agents_completed"] == []


# --- End-to-end through the real compiled graph ---------------------------

VALID_PY = "def greet():\n    return 'hello'\n"

# >= 25 branches trips CodeMetricsTool's "high" complexity threshold, which is
# what makes the DispatcherAgent ask for deep analysis and therefore gives the
# OrchestratorAgent two independent sub-tasks to fan out.
HIGH_COMPLEXITY_PY = "def f(x):\n" + "".join(
    f"    if x == {i}:\n        return {i}\n" for i in range(30)
)


class OrchestrationLLM:
    """A stub covering every agent prompt on the orchestrated path."""

    async def call_llm(self, prompt, system_prompt="", fmt=None, model=None) -> str:
        if prompt.startswith("Analyze this"):
            return json.dumps(
                {
                    "deprecated_patterns": [],
                    "migration_challenges": [],
                    "key_constructs": ["f"],
                    "summary": "Stub analysis.",
                }
            )
        if "deep analysis" in prompt or "expert code analyst" in prompt:
            return json.dumps(
                {
                    "complex_constructs": ["long if-chain"],
                    "stdlib_dependencies": [],
                    "inheritance_depth": 0,
                    "design_patterns": [],
                    "breaking_changes": [],
                    "recommended_strategy": "mechanical",
                }
            )
        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps({"plan_summary": "plan", "steps": [], "risk_areas": []})
        if "failed validation" in prompt:
            return VALID_PY
        return json.dumps({"plan_summary": "m", "migrated_code": VALID_PY})

    def extract_json(self, raw):
        return json.loads(raw)


def _e2e_state(**overrides) -> dict:
    base = _graph_state(
        source_code=HIGH_COMPLEXITY_PY,
        migration_type="upgrade_version",
        parallel_enabled=True,
        max_parallel_tasks=4,
        parallel_tasks=[],
        subgraph_results=[],
        retry_count=0,
        max_retries=1,
        reretrieval_count=0,
        max_reretrievals=0,
    )
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_end_to_end_fan_out_runs_both_branches_and_merges():
    """High complexity -> orchestrator plans 2 tasks -> both subgraphs run + merge."""
    from graph import nodes
    from graph.migration_graph import build_migration_graph

    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(OrchestrationLLM())
    try:
        result = await build_migration_graph().ainvoke(_e2e_state())
    finally:
        nodes.set_llm_client(None)

    # The orchestrator found concurrent work and the fan-out actually ran.
    assert result["route_plan"]["parallelizable"] is True
    assert {r["task"] for r in result["subgraph_results"]} == {"analysis", "retrieval"}
    assert all(r["ok"] for r in result["subgraph_results"])
    # Each record names only what that branch contributed, not the shared
    # history it inherited from before the fan-out.
    by_task = {r["task"]: r for r in result["subgraph_results"]}
    assert by_task["analysis"]["agents"] == ["DeepAnalyzerAgent"]
    assert by_task["retrieval"]["agents"] == ["RetrieverAgent"]

    # Both branches' agents survived the merge, each recorded exactly once.
    agents = result["agents_completed"]
    assert agents.count("DeepAnalyzerAgent") == 1
    assert agents.count("RetrieverAgent") == 1
    # And the branch that ran before the fan-out was not duplicated by it.
    assert agents.count("AnalyzerAgent") == 1

    # The key-wise metrics merge kept the deep-analysis enrichment alongside the
    # base metrics the AnalyzerAgent wrote before the split.
    assert "deep_analysis" in result["code_metrics"]
    assert result["code_metrics"]["complexity"] == "high"

    # The run still completed normally past the merge.
    assert result["migrated_code"].strip() == VALID_PY.strip()
    assert result["validation_result"]["valid"] is True
    assert not result["errors"]


@pytest.mark.asyncio
async def test_end_to_end_disabled_takes_the_original_sequential_path():
    """With the fan-out off, the graph behaves exactly as it did before."""
    from graph import nodes
    from graph.migration_graph import build_migration_graph

    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(OrchestrationLLM())
    try:
        result = await build_migration_graph().ainvoke(
            _e2e_state(parallel_enabled=False)
        )
    finally:
        nodes.set_llm_client(None)

    # The decomposition still found independent work; the kill switch is what
    # stopped it being executed that way, and nothing fanned out.
    assert result["route_plan"]["parallelizable"] is True
    assert result["subgraph_results"] == []
    # Same agents, same outcome — just reached sequentially.
    assert result["agents_completed"].count("DeepAnalyzerAgent") == 1
    assert result["agents_completed"].count("RetrieverAgent") == 1
    assert "deep_analysis" in result["code_metrics"]
    assert result["migrated_code"].strip() == VALID_PY.strip()
    assert not result["errors"]


@pytest.mark.asyncio
async def test_end_to_end_low_complexity_skips_the_fan_out():
    """No deep analysis -> one sub-task -> sequential, no subgraph overhead."""
    from graph import nodes
    from graph.migration_graph import build_migration_graph

    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(OrchestrationLLM())
    try:
        result = await build_migration_graph().ainvoke(
            _e2e_state(source_code="x = 1\n")
        )
    finally:
        nodes.set_llm_client(None)

    assert result["subgraph_results"] == []
    assert "DeepAnalyzerAgent" not in result["agents_completed"]
    assert "RetrieverAgent" in result["agents_completed"]
    assert not result["errors"]


class TestGraphWiring:
    def test_graph_compiles_with_orchestration_nodes(self):
        from graph.migration_graph import build_migration_graph

        nodes = build_migration_graph().get_graph().nodes
        assert "orchestrate" in nodes
        # One node serving every fanned-out sub-task; orchestrate_condition
        # names it in each Send.
        assert "branch" in nodes

    def test_sequential_path_is_preserved(self):
        # Orchestration reshapes only the pre-planning phase; the nodes carrying
        # the fix / reflect / re-retrieval loops must still be present.
        from graph.migration_graph import build_migration_graph

        nodes = build_migration_graph().get_graph().nodes
        for node in ("deep_analyze", "retrieve", "plan", "migrate", "reflect", "fix"):
            assert node in nodes

    def test_new_state_fields_are_declared(self):
        from graph.state import GraphState

        for key in ("parallel_tasks", "subgraph_results", "parallel_enabled"):
            assert key in GraphState.__annotations__
