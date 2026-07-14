"""Offline tests for the Phase 3 LangGraph migration graph.

These exercise the graph structure and the validate -> fix -> migrate -> validate
retry loop without a running Ollama/validator service by injecting a stub LLM
client and relying on the offline Python syntax validator (``ast.parse``).

Run: pytest tests_graph.py -v
Or:  python tests_graph.py
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from graph import nodes
from graph.cicd_graph import build_cicd_graph
from graph.conditions import complexity_condition, migrate_condition, validate_condition
from graph.migration_graph import build_migration_graph

VALID_PY = "def greet():\n    return 'hello'\n"
INVALID_PY = "def broken(:\n    pass\n"


class StubLLM:
    """Scripted LLM: routes responses by prompt content.

    - Planner prompts -> a plan JSON object.
    - Fixer prompts    -> raw corrected source code.
    - Migrator prompts -> JSON; the first N migrator calls return invalid code
      (controlled by ``invalid_migrations``), the rest return valid code.
    """

    def __init__(self, invalid_migrations: int = 1):
        self.invalid_migrations = invalid_migrations
        self.migrator_calls = 0
        self.calls: list[str] = []

    async def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        self.calls.append(prompt)

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
    for expected in {"analyze", "deep_analyze", "plan", "migrate", "validate", "fix"}:
        assert expected in node_names


def test_cicd_graph_compiles():
    app = build_cicd_graph()
    node_names = set(getattr(app, "nodes", {}) or {})
    for expected in {"check_pr", "create_pr", "wait_ci", "auto_merge"}:
        assert expected in node_names


def test_complexity_condition_routes_both_branches():
    assert complexity_condition({"code_metrics": {"complexity": "high"}}) == "deep_analyze"
    assert complexity_condition({"code_metrics": {"complexity": "medium"}}) == "retrieve"
    assert complexity_condition({"code_metrics": {"complexity": "low"}}) == "retrieve"
    assert complexity_condition({}) == "retrieve"  # missing metrics default


def test_migrate_condition_routes_empty_code_to_end():
    assert migrate_condition({"migrated_code": ""}) == "end"
    assert migrate_condition({}) == "end"
    assert migrate_condition({"migrated_code": "x = 1"}) == "validate"


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
        async def call_llm(self, prompt: str, system_prompt: str = "") -> str:
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

    assert result["migrated_code"] == ""
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
