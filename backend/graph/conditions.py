"""Edge routing logic for the migration graph.

These functions are used as LangGraph conditional-edge selectors: they inspect
the current state and return the name of the branch to take.
"""

import logging

log = logging.getLogger("CodeMigrateAI.GraphConditions")


def complexity_condition(state: dict) -> str:
    """Route: low/medium -> standard path, high -> deep analysis."""
    metrics = state.get("code_metrics") or {}
    complexity = metrics.get("complexity", "low")
    route = "deep_analyze" if complexity == "high" else "retrieve"
    log.info("Complexity: %s -> %s", complexity, route)
    return route


def migrate_condition(state: dict) -> str:
    """Route: MigratorAgent produced code -> validate, produced nothing -> end.

    MigratorAgent's own failures (e.g. the LLM call raising) are caught by
    BaseAgent.__call__ and recorded into ``errors`` rather than raised, so the
    graph would otherwise keep running ValidatorAgent/FixerAgent against an
    empty ``migrated_code`` string. Ending early here mirrors the previous
    linear pipeline's explicit stop-on-MigratorAgent-failure behavior.
    """
    if not state.get("migrated_code"):
        log.info("MigratorAgent produced no code -> end")
        return "end"
    return "validate"


def validate_condition(state: dict) -> str:
    """Route: pass -> end, fail+retries left -> fix, fail+exhausted -> end.

    ``retry_count`` is advanced by the ``fix`` node (see ``nodes.fix_node``), so
    each failed validation that routes to ``fix`` consumes one retry and the
    loop terminates once ``retry_count`` reaches ``max_retries``.
    """
    validation = state.get("validation_result") or {}
    valid = validation.get("valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if valid:
        log.info("Validation PASSED -> end")
        return "end"
    if retry_count < max_retries:
        log.info(
            "Validation FAILED (retry %d/%d) -> fix", retry_count + 1, max_retries
        )
        return "fix"

    log.info("Validation FAILED (exhausted %d retries) -> end (best effort)", max_retries)
    # Preserve the best effort code before ending (also set defensively in the
    # fix node, since edge selectors are not guaranteed to persist mutations).
    if state.get("migrated_code") and not state.get("best_effort_code"):
        state["best_effort_code"] = state["migrated_code"]
    return "end"
