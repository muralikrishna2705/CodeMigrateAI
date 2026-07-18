"""Edge routing logic for the migration graph.

These functions are used as LangGraph conditional-edge selectors: they inspect
the current state and return the name of the branch to take.
"""

import logging

log = logging.getLogger("CodeMigrateAI.GraphConditions")


def dispatch_condition(state: dict) -> str:
    """Route after dispatch: honor the DispatcherAgent's plan, else fall back.

    The ``dispatch`` node (DispatcherAgent) writes ``route_plan`` with its routing
    decision, so routing is data-driven and LLM-pluggable rather than a hardcoded
    complexity check here. If the plan is absent (e.g. a unit test invoking this
    condition directly), fall back to the original complexity heuristic.
    """
    plan = state.get("route_plan") or {}
    if "deep_analyze" in plan:
        route = "deep_analyze" if plan["deep_analyze"] else "retrieve"
        log.info("Dispatch plan: deep_analyze=%s -> %s", plan["deep_analyze"], route)
        return route

    complexity = (state.get("code_metrics") or {}).get("complexity", "low")
    route = "deep_analyze" if complexity == "high" else "retrieve"
    log.info("Dispatch fallback (no plan): complexity=%s -> %s", complexity, route)
    return route


# Back-compat alias: the routing decision moved from an analyze-time complexity
# check to the DispatcherAgent, but the branch semantics are identical.
complexity_condition = dispatch_condition


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
    # Adaptive RAG: the MigratorAgent enqueues ungrounded imports as targeted
    # retrieval queries. While any are pending and the re-retrieval budget is not
    # spent, loop back through retrieve -> plan -> migrate for better grounding.
    requests = state.get("retrieval_requests") or []
    reretrieval_count = state.get("reretrieval_count", 0)
    max_reretrievals = state.get("max_reretrievals", 0)
    if requests and reretrieval_count < max_reretrievals:
        log.info(
            "Ungrounded imports pending (%d) -> retrieve (re-retrieval %d/%d)",
            len(requests),
            reretrieval_count + 1,
            max_reretrievals,
        )
        return "retrieve"
    return "validate"


def reflect_condition(state: dict) -> str:
    """Route after the reflect node: pass -> validate, low-confidence -> re-migrate.

    The ReflectorAgent writes ``reflection_recommendation`` (``"pass"`` /
    ``"re-generate"`` / ``"gather-more-info"``). A non-pass verdict routes back to
    ``migrate`` so the MigratorAgent can regenerate with the reflection feedback —
    but only while the reflection budget remains, mirroring how ``retry_count``
    bounds the fix loop. Every other case (a passing verdict, reflection disabled
    so the node was skipped and the field defaults to ``"pass"``, or the budget
    spent) proceeds to validation.

    The budget is consumed by ``migrate_node`` after it regenerates (it detects a
    reflection-driven re-entry the way ``retrieve_node`` detects a re-retrieval),
    so this selector stays a pure read of the state.
    """
    recommendation = state.get("reflection_recommendation") or "pass"
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", 0)

    if recommendation != "pass" and reflection_count < max_reflections:
        log.info(
            "Reflection verdict %r -> re-migrate (reflection %d/%d)",
            recommendation,
            reflection_count + 1,
            max_reflections,
        )
        return "re_migrate"

    if recommendation != "pass":
        log.info(
            "Reflection verdict %r but budget spent (%d/%d) -> validate",
            recommendation,
            reflection_count,
            max_reflections,
        )
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
