"""Tests for dynamic orchestration: decomposition, fan-out, and fan-in merge.

Everything runs offline against scripted stub LLMs — no Ollama/validator service —
including an end-to-end fan-out through the real compiled graph and subgraphs.
"""

import asyncio
import json

import pytest

from agents.orchestrator_agent import OrchestratorAgent
from graph.conditions import orchestrate_condition
from graph.merge import merge_results
from graph.parallel import BranchResult, run_parallel
from graph.subgraphs import SUBGRAPH_TASKS, resolve_tasks
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
    """A compiled-graph stand-in: mutates the state it is handed and returns it."""

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
        if self.mutate:
            self.mutate(state)
        return state


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
    def test_parallel_tasks_route_to_parallel(self):
        state = {"parallel_enabled": True, "parallel_tasks": ["analysis", "retrieval"]}
        assert orchestrate_condition(state) == "parallel"

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


class TestRunParallel:
    @pytest.mark.asyncio
    async def test_branches_run_concurrently(self):
        # Two 50ms branches must overlap; serialized they would take ~100ms.
        tasks = [
            ("a", _StubSubgraph("a", delay=0.05)),
            ("b", _StubSubgraph("b", delay=0.05)),
        ]
        started = asyncio.get_event_loop().time()
        await run_parallel(tasks, _graph_state(), max_concurrency=2)
        elapsed = asyncio.get_event_loop().time() - started

        assert elapsed < 0.09

    @pytest.mark.asyncio
    async def test_semaphore_bounds_concurrency(self):
        active = 0
        peak = 0

        class _Counting(_StubSubgraph):
            async def ainvoke(self, state):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return state

        tasks = [(f"t{i}", _Counting(f"t{i}")) for i in range(6)]
        await run_parallel(tasks, _graph_state(), max_concurrency=2)

        assert peak <= 2

    @pytest.mark.asyncio
    async def test_branches_get_isolated_state_copies(self):
        def mutate_a(state):
            state["rag_context"] = "from-a"

        def mutate_b(state):
            state["inline_plan"] = "from-b"

        sub_a = _StubSubgraph("a", mutate=mutate_a)
        sub_b = _StubSubgraph("b", mutate=mutate_b)
        base = _graph_state()
        await run_parallel([("a", sub_a), ("b", sub_b)], base, max_concurrency=2)

        # Neither branch saw the other's write, and the caller's state is intact.
        assert "rag_context" not in base
        assert "inline_plan" not in base
        assert sub_a.seen_states[0] is not base

    @pytest.mark.asyncio
    async def test_failed_branch_does_not_abort_siblings(self):
        sub_ok = _StubSubgraph("ok", mutate=lambda s: s.update(rag_context="ctx"))
        sub_bad = _StubSubgraph("bad", raises=RuntimeError("boom"))
        branches = await run_parallel(
            [("bad", sub_bad), ("ok", sub_ok)], _graph_state(), max_concurrency=2
        )

        by_task = {b.task: b for b in branches}
        assert by_task["bad"].ok is False
        assert "boom" in by_task["bad"].error
        assert by_task["ok"].ok is True

    @pytest.mark.asyncio
    async def test_results_keep_task_order_not_completion_order(self):
        slow = _StubSubgraph("slow", delay=0.03)
        fast = _StubSubgraph("fast")
        branches = await run_parallel(
            [("slow", slow), ("fast", fast)], _graph_state(), max_concurrency=2
        )
        assert [b.task for b in branches] == ["slow", "fast"]

    @pytest.mark.asyncio
    async def test_empty_task_list_is_a_noop(self):
        assert await run_parallel([], _graph_state()) == []

    @pytest.mark.asyncio
    async def test_zero_concurrency_does_not_deadlock(self):
        branches = await run_parallel(
            [("a", _StubSubgraph("a"))], _graph_state(), max_concurrency=0
        )
        assert branches[0].ok


class TestMergeResults:
    def _branch(self, task, state):
        return BranchResult(task, state=state)

    def test_report_deltas_are_appended_without_duplicating_history(self):
        base = _graph_state(reports=[{"agent": "AnalyzerAgent"}])
        a = _graph_state(
            reports=[{"agent": "AnalyzerAgent"}, {"agent": "DeepAnalyzerAgent"}]
        )
        b = _graph_state(
            reports=[{"agent": "AnalyzerAgent"}, {"agent": "RetrieverAgent"}]
        )
        merged = merge_results(
            base, [self._branch("analysis", a), self._branch("retrieval", b)]
        )

        assert [r["agent"] for r in merged["reports"]] == [
            "AnalyzerAgent",
            "DeepAnalyzerAgent",
            "RetrieverAgent",
        ]

    def test_agents_completed_and_errors_merge_the_same_way(self):
        base = _graph_state(agents_completed=["AnalyzerAgent"], errors=[])
        a = _graph_state(
            agents_completed=["AnalyzerAgent", "DeepAnalyzerAgent"], errors=["e1"]
        )
        b = _graph_state(
            agents_completed=["AnalyzerAgent", "RetrieverAgent"], errors=[]
        )
        merged = merge_results(
            base, [self._branch("analysis", a), self._branch("retrieval", b)]
        )

        assert merged["agents_completed"] == [
            "AnalyzerAgent",
            "DeepAnalyzerAgent",
            "RetrieverAgent",
        ]
        assert merged["errors"] == ["e1"]

    def test_code_metrics_merge_key_wise(self):
        # The real conflict: both branches carry a copy of the base metrics and
        # one enriches it. A whole-dict overwrite would drop deep_analysis.
        base = _graph_state(code_metrics={"complexity": "high", "loc": 100})
        a = _graph_state(
            code_metrics={
                "complexity": "high",
                "loc": 100,
                "deep_analysis": {"patterns": ["visitor"]},
            }
        )
        b = _graph_state(code_metrics={"complexity": "high", "loc": 100})
        merged = merge_results(
            base, [self._branch("analysis", a), self._branch("retrieval", b)]
        )

        assert merged["code_metrics"]["deep_analysis"] == {"patterns": ["visitor"]}
        assert merged["code_metrics"]["loc"] == 100

    def test_scalar_from_one_branch_wins(self):
        base = _graph_state(rag_context="")
        a = _graph_state(rag_context="")
        b = _graph_state(rag_context="retrieved docs")
        merged = merge_results(
            base, [self._branch("analysis", a), self._branch("retrieval", b)]
        )
        assert merged["rag_context"] == "retrieved docs"

    def test_conflicting_scalar_resolves_by_task_order(self):
        base = _graph_state(rag_context="")
        a = _graph_state(rag_context="from-a")
        b = _graph_state(rag_context="from-b")
        merged = merge_results(
            base, [self._branch("first", a), self._branch("second", b)]
        )
        # Deterministic regardless of completion order.
        assert merged["rag_context"] == "from-a"

    def test_counters_take_the_max(self):
        base = _graph_state(reretrieval_count=0)
        a = _graph_state(reretrieval_count=0)
        b = _graph_state(reretrieval_count=1)
        merged = merge_results(base, [self._branch("a", a), self._branch("b", b)])
        assert merged["reretrieval_count"] == 1

    def test_failed_branch_contributes_nothing_but_is_recorded(self):
        base = _graph_state(reports=[{"agent": "AnalyzerAgent"}])
        ok = _graph_state(
            reports=[{"agent": "AnalyzerAgent"}, {"agent": "RetrieverAgent"}]
        )
        merged = merge_results(
            base,
            [
                BranchResult("analysis", error="boom"),
                self._branch("retrieval", ok),
            ],
        )

        assert [r["agent"] for r in merged["reports"]] == [
            "AnalyzerAgent",
            "RetrieverAgent",
        ]
        results = {r["task"]: r for r in merged["subgraph_results"]}
        assert results["analysis"]["ok"] is False
        assert results["analysis"]["error"] == "boom"
        assert results["retrieval"]["ok"] is True

    def test_all_branches_failing_degrades_to_base_state(self):
        base = _graph_state(reports=[{"agent": "AnalyzerAgent"}], rag_context="")
        merged = merge_results(
            base,
            [BranchResult("analysis", error="a"), BranchResult("retrieval", error="b")],
        )
        assert merged["reports"] == [{"agent": "AnalyzerAgent"}]
        assert len(merged["subgraph_results"]) == 2

    def test_base_state_is_not_mutated(self):
        base = _graph_state(reports=[])
        branch = _graph_state(reports=[{"agent": "RetrieverAgent"}])
        merge_results(base, [self._branch("retrieval", branch)])
        assert base["reports"] == []


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


class TestParallelNode:
    @pytest.mark.asyncio
    async def test_self_skips_when_disabled(self):
        from graph.nodes import parallel_node

        state = _graph_state(
            parallel_enabled=False, parallel_tasks=["analysis", "retrieval"]
        )
        assert await parallel_node(state) is state

    @pytest.mark.asyncio
    async def test_self_skips_without_concurrent_work(self):
        from graph.nodes import parallel_node

        state = _graph_state(parallel_enabled=True, parallel_tasks=["retrieval"])
        assert await parallel_node(state) is state

    @pytest.mark.asyncio
    async def test_self_skips_on_unseeded_state(self):
        from graph.nodes import parallel_node

        state = _graph_state()
        assert await parallel_node(state) is state


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
        assert "parallel" in nodes

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
