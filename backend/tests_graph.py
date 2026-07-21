"""Offline tests for the Phase 3 LangGraph migration graph.

These exercise the graph structure and the validate -> fix -> migrate -> validate
retry loop without a running Ollama/validator service by injecting a stub LLM
client and relying on the offline Python syntax validator (``ast.parse``).

Run: pytest tests_graph.py -v
Or:  python tests_graph.py
"""

import asyncio
import json

import pytest

from graph import nodes
from graph.cicd_graph import build_cicd_graph
from graph.conditions import (
    complexity_condition,
    dispatch_condition,
    migrate_condition,
    validate_condition,
)
from graph.migration_graph import build_migration_graph

VALID_PY = "def greet():\n    return 'hello'\n"
INVALID_PY = "def broken(:\n    pass\n"


class StubLLM:
    """Scripted LLM: routes responses by prompt content.

    - Analyzer prompts -> a semantic-analysis JSON object.
    - Planner prompts  -> a plan JSON object.
    - Fixer prompts    -> raw corrected source code.
    - Migrator prompts -> JSON; the first N migrator calls return invalid code
      (controlled by ``invalid_migrations``), the rest return valid code.

    Every caller must be routed explicitly, because the migrator branch is the
    fall-through and it is *stateful*: an unrouted prompt silently consumes the
    ``invalid_migrations`` budget, so the migrator returns valid code earlier
    than the test intends and the fix loop never runs. Add a branch here when an
    agent starts calling the LLM.
    """

    def __init__(self, invalid_migrations: int = 1):
        self.invalid_migrations = invalid_migrations
        self.migrator_calls = 0
        self.calls: list[str] = []

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> str:
        self.calls.append(prompt)

        # AnalyzerAgent: keyed on ANALYZER_PROMPT's opening line. Matching on a
        # schema key like "deprecated_patterns" would be wrong — the migrator
        # prompt embeds code_metrics, so it contains those key names too.
        if prompt.startswith("Analyze this"):
            return json.dumps(
                {
                    "deprecated_patterns": [],
                    "migration_challenges": [],
                    "key_constructs": ["print statement"],
                    "summary": "Stub analysis.",
                }
            )

        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps(
                {"plan_summary": "Upgrade the code.", "steps": [], "risk_areas": []}
            )

        if "failed validation" in prompt:
            # FixerAgent: return corrected source directly (no fences).
            return VALID_PY

        # MigratorAgent
        self.migrator_calls += 1
        code = INVALID_PY if self.migrator_calls <= self.invalid_migrations else VALID_PY
        return json.dumps({"plan_summary": "Migrated.", "migrated_code": code})

    def extract_json(self, raw_text: str) -> dict:
        return json.loads(raw_text)


def _initial_state() -> dict:
    return {
        "source_code": "print('hello')",
        "source_language": "python",
        "source_version": "2.7",
        "target_language": "python",
        "target_version": "3.12",
        "migration_type": "upgrade_version",
        "retry_count": 0,
        "max_retries": 2,
        "best_effort_code": "",
        "reports": [],
        "errors": [],
        "agents_completed": [],
    }


def test_graph_compiles_with_all_nodes():
    app = build_migration_graph()
    node_names = set(getattr(app, "nodes", {}) or {})
    for expected in {
        "analyze",
        "dispatch",
        "deep_analyze",
        "retrieve",
        "plan",
        "migrate",
        "validate",
        "fix",
        "service_validate",
        "observe",
    }:
        assert expected in node_names


@pytest.mark.asyncio
async def test_open_circuit_skips_agent_node():
    """A node whose circuit is open is skipped without running the agent."""
    from graph.nodes import _make_node
    from runtime import agent_recovery

    agent_recovery.reset()
    for _ in range(3):  # trip the breaker
        agent_recovery.record_failure("AnalyzerAgent")
    assert agent_recovery.is_circuit_open("AnalyzerAgent")

    node = _make_node("AnalyzerAgent")
    result = await node(_initial_state())
    # Skipped: the agent never ran, so it isn't recorded as completed.
    assert "AnalyzerAgent" not in result.get("agents_completed", [])
    agent_recovery.reset()


def test_cicd_graph_compiles():
    app = build_cicd_graph()
    node_names = set(getattr(app, "nodes", {}) or {})
    for expected in {"check_pr", "create_pr", "wait_ci", "auto_merge"}:
        assert expected in node_names


def test_complexity_condition_routes_both_branches():
    # No route_plan -> falls back to the complexity heuristic (back-compat alias).
    assert complexity_condition({"code_metrics": {"complexity": "high"}}) == "deep_analyze"
    assert complexity_condition({"code_metrics": {"complexity": "medium"}}) == "retrieve"
    assert complexity_condition({"code_metrics": {"complexity": "low"}}) == "retrieve"
    assert complexity_condition({}) == "retrieve"  # missing metrics default


def test_dispatch_condition_honors_route_plan():
    # The dispatcher's plan wins over the raw complexity metric.
    assert (
        dispatch_condition(
            {"route_plan": {"deep_analyze": True}, "code_metrics": {"complexity": "low"}}
        )
        == "deep_analyze"
    )
    assert (
        dispatch_condition(
            {"route_plan": {"deep_analyze": False}, "code_metrics": {"complexity": "high"}}
        )
        == "retrieve"
    )
    # Absent plan -> complexity fallback.
    assert dispatch_condition({"code_metrics": {"complexity": "high"}}) == "deep_analyze"
    assert dispatch_condition({}) == "retrieve"


@pytest.mark.asyncio
async def test_dispatcher_agent_writes_route_plan():
    """DispatcherAgent turns detected complexity into a consumable route plan."""
    from models.state import MigrationState
    from runtime.agent_dispatcher import DispatcherAgent

    def _state(complexity: str) -> MigrationState:
        s = MigrationState(
            source_code="x",
            source_language="python",
            source_version="3.8",
            target_language="python",
            target_version="3.12",
        )
        s.code_metrics = {"complexity": complexity}
        return s

    agent = DispatcherAgent(None)

    high = await agent(_state("high"))
    assert high.route_plan["deep_analyze"] is True
    assert "DeepAnalyzerAgent" in high.route_plan["sequence"]

    low = await agent(_state("low"))
    assert low.route_plan["deep_analyze"] is False
    assert "DeepAnalyzerAgent" not in low.route_plan["sequence"]


def test_migrate_condition_routes_empty_code_to_end():
    assert migrate_condition({"migrated_code": ""}) == "end"
    assert migrate_condition({}) == "end"
    assert migrate_condition({"migrated_code": "x = 1"}) == "validate"


def test_migrate_condition_routes_to_retrieve_on_ungrounded_with_budget():
    assert (
        migrate_condition(
            {
                "migrated_code": "x = 1",
                "retrieval_requests": ["numpy"],
                "reretrieval_count": 0,
                "max_reretrievals": 1,
            }
        )
        == "retrieve"
    )


def test_migrate_condition_validates_when_reretrieval_budget_spent():
    assert (
        migrate_condition(
            {
                "migrated_code": "x = 1",
                "retrieval_requests": ["numpy"],
                "reretrieval_count": 1,
                "max_reretrievals": 1,
            }
        )
        == "validate"
    )


def test_validate_condition_routes_pass_fix_and_exhausted():
    # Passing validation ends the graph.
    assert validate_condition({"validation_result": {"valid": True}}) == "end"
    # Failing with retries remaining routes to fix.
    assert (
        validate_condition(
            {"validation_result": {"valid": False}, "retry_count": 0, "max_retries": 2}
        )
        == "fix"
    )
    # Failing with retries exhausted ends (and preserves best effort).
    exhausted = {
        "validation_result": {"valid": False},
        "retry_count": 2,
        "max_retries": 2,
        "migrated_code": "x = 1",
    }
    assert validate_condition(exhausted) == "end"
    assert exhausted["best_effort_code"] == "x = 1"


@pytest.mark.asyncio
async def test_retry_loop_recovers_after_fix():
    """Invalid migration -> fix -> re-migrate -> valid -> end."""
    stub = StubLLM(invalid_migrations=1)
    nodes.set_llm_client(stub)
    try:
        app = build_migration_graph()
        result = await app.ainvoke(_initial_state())
    finally:
        nodes.set_llm_client(None)

    # Ran the loop exactly once (one fix attempt) before passing.
    assert result["retry_count"] == 1
    assert result["validation_result"]["valid"] is True
    assert result["migrated_code"].strip() == VALID_PY.strip()
    # Complexity was low -> deep_analyze skipped, plan ran.
    assert "AnalyzerAgent" in result["agents_completed"]
    assert "PlannerAgent" in result["agents_completed"]
    assert "DeepAnalyzerAgent" not in result["agents_completed"]


@pytest.mark.asyncio
async def test_migrator_failure_ends_without_validate_or_fix():
    """MigratorAgent raising (LLM unreachable) ends the graph, skipping validate/fix."""

    class FailingLLM:
        async def call_llm(
            self, prompt: str, system_prompt: str = "", fmt: str | None = None
        ) -> str:
            if "MIGRATION PLANNING TASK" in prompt:
                return json.dumps({"plan_summary": "plan", "steps": [], "risk_areas": []})
            raise ConnectionError("LLM unreachable")

        def extract_json(self, raw_text: str) -> dict:
            return json.loads(raw_text)

    nodes.set_llm_client(FailingLLM())
    try:
        app = build_migration_graph()
        result = await app.ainvoke(_initial_state())
    finally:
        nodes.set_llm_client(None)

    # Absent, not empty. Nodes return only what they changed, so a channel no
    # node ever wrote holds no value at all — and the migrator failed before it
    # produced any code. The app seeds every key up front (see
    # Pipeline._run_graph), so this shape is specific to invoking the compiled
    # graph directly with a minimal state.
    assert result.get("migrated_code", "") == ""
    assert "ValidatorAgent" not in result["agents_completed"]
    assert "FixerAgent" not in result["agents_completed"]
    assert any("MigratorAgent" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_retry_loop_exhausts_and_keeps_best_effort():
    """Always-invalid migration exhausts max_retries and ends on best effort."""
    stub = StubLLM(invalid_migrations=99)
    nodes.set_llm_client(stub)
    try:
        app = build_migration_graph()
        result = await app.ainvoke(_initial_state())
    finally:
        nodes.set_llm_client(None)

    assert result["retry_count"] == 2  # == max_retries
    assert result["validation_result"]["valid"] is False
    assert result["best_effort_code"]  # preserved before ending


@pytest.mark.asyncio
async def test_retrieve_node_counts_and_clears_reretrieval():
    """A re-retrieval pass advances the counter and drains the request queue."""
    nodes.set_rag_pipeline(None)  # RetrieverAgent skips; wrapper still counts
    nodes.set_llm_client(StubLLM())
    try:
        state = _initial_state()
        state["retrieval_requests"] = ["numpy"]
        state["reretrieval_count"] = 0
        result = await nodes.retrieve_node(state)
    finally:
        nodes.set_llm_client(None)

    assert result["reretrieval_count"] == 1
    assert result["retrieval_requests"] == []


@pytest.mark.asyncio
async def test_adaptive_reretrieval_loop_on_ungrounded_imports():
    """Ungrounded import -> retrieve -> re-migrate clean, bounded to one loop."""

    class ReretrieveLLM:
        def __init__(self):
            self.migrator_calls = 0

        async def call_llm(self, prompt, system_prompt="", fmt=None, **kwargs):
            # Routed before the stateful migrator fall-through, so the analyzer's
            # semantic call cannot consume one of the scripted migrations.
            if prompt.startswith("Analyze this"):
                return json.dumps(
                    {
                        "deprecated_patterns": [],
                        "migration_challenges": [],
                        "key_constructs": [],
                        "summary": "Stub analysis.",
                    }
                )
            if "MIGRATION PLANNING TASK" in prompt:
                return json.dumps(
                    {"plan_summary": "plan", "steps": [], "risk_areas": []}
                )
            if "failed validation" in prompt:
                return VALID_PY
            self.migrator_calls += 1
            # First migration invents a package; the second (post re-retrieval)
            # is clean and grounded.
            code = (
                "import nonexistent_pkg\n\ndef greet():\n    return 'hi'\n"
                if self.migrator_calls == 1
                else VALID_PY
            )
            return json.dumps({"plan_summary": "m", "migrated_code": code})

        def extract_json(self, raw_text):
            return json.loads(raw_text)

    nodes.set_rag_pipeline(None)
    nodes.set_llm_client(ReretrieveLLM())
    try:
        app = build_migration_graph()
        state = _initial_state()
        state["max_reretrievals"] = 1
        result = await app.ainvoke(state)
    finally:
        nodes.set_llm_client(None)

    assert result["reretrieval_count"] == 1
    assert result["migrated_code"].strip() == VALID_PY.strip()
    # The observer terminal node ran, so its agent is recorded.
    assert "ObserverAgent" in result["agents_completed"]


async def _main() -> None:
    test_graph_compiles_with_all_nodes()
    test_cicd_graph_compiles()
    test_complexity_condition_routes_both_branches()
    test_migrate_condition_routes_empty_code_to_end()
    test_validate_condition_routes_pass_fix_and_exhausted()
    await test_retry_loop_recovers_after_fix()
    await test_migrator_failure_ends_without_validate_or_fix()
    await test_retry_loop_exhausts_and_keeps_best_effort()
    print("All graph verification checks passed.")


if __name__ == "__main__":
    asyncio.run(_main())
