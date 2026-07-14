"""CI/CD wrapper graph (Phase 4 skeleton).

This is an intentionally thin skeleton that lays out the node/edge structure for
an automated repository migration workflow::

    checkout ──▶ migrate_repo ──▶ test ──(pass)──▶ open_pr ──▶ END
                                       └──(fail)──────────────▶ END

Each node currently records progress and passes state through. Phase 4 fills in
the real integrations (git checkout, per-file migration via the migration graph,
test runners, and the GitHub PR API).
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

log = logging.getLogger("CodeMigrateAI.CICDGraph")


class CICDState(TypedDict, total=False):
    """State schema for the CI/CD migration workflow."""

    repo_url: str
    branch: str
    files: list[dict]  # [{path, source_code, source_language, source_version}]
    migration_config: dict  # {target_language, target_version, ...}
    migrated_files: list[dict]
    test_results: Optional[dict]
    pr_url: str
    status: str
    errors: list[str]


async def checkout_node(state: dict) -> dict:
    """Clone/checkout the target repository (skeleton)."""
    log.info(
        "[cicd] checkout %s@%s (skeleton)",
        state.get("repo_url"),
        state.get("branch"),
    )
    state.setdefault("files", [])
    state["status"] = "checked_out"
    return state


async def migrate_repo_node(state: dict) -> dict:
    """Run the migration graph over each source file (skeleton).

    Phase 4: for each file in ``state['files']``, build the migration graph
    (:func:`graph.migration_graph.build_migration_graph`) and collect the
    migrated code. Left unwired here so the skeleton has no hard dependency on a
    running LLM.
    """
    files = state.get("files", [])
    log.info("[cicd] migrate %d file(s) (skeleton)", len(files))
    state["migrated_files"] = list(files)
    state["status"] = "migrated"
    return state


async def test_node(state: dict) -> dict:
    """Run the project's test suite against the migrated code (skeleton)."""
    log.info("[cicd] run tests (skeleton)")
    state["test_results"] = {"passed": True, "skipped": True}
    state["status"] = "tested"
    return state


async def open_pr_node(state: dict) -> dict:
    """Open a pull request with the migrated changes (skeleton)."""
    log.info("[cicd] open pull request (skeleton)")
    state["pr_url"] = ""
    state["status"] = "pr_opened"
    return state


def tests_condition(state: dict) -> str:
    """Route: tests pass -> open PR, tests fail -> end."""
    results = state.get("test_results") or {}
    return "open_pr" if results.get("passed", False) else "end"


def build_cicd_graph():
    """Construct and compile the CI/CD workflow graph (Phase 4 skeleton)."""
    workflow = StateGraph(CICDState)

    workflow.add_node("checkout", checkout_node)
    workflow.add_node("migrate", migrate_repo_node)
    workflow.add_node("test", test_node)
    workflow.add_node("open_pr", open_pr_node)

    workflow.set_entry_point("checkout")
    workflow.add_edge("checkout", "migrate")
    workflow.add_edge("migrate", "test")
    workflow.add_conditional_edges(
        "test",
        tests_condition,
        {"open_pr": "open_pr", "end": END},
    )
    workflow.add_edge("open_pr", END)

    app = workflow.compile()
    log.info("CI/CD graph compiled (Phase 4 skeleton)")
    return app
