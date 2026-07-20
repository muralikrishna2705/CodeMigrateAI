"""Build and compile the migration StateGraph.

Flow::

    analyze ─▶ dispatch ─▶ orchestrate ─(parallel)─▶ parallel ─▶ plan ─▶ migrate ─▶ reflect ─▶ validate
                              │                    [analysis ‖         ▲            │            │
                              │                     retrieval]         │            │            │
                              ├──(deep)──▶ deep_analyze ─▶ retrieve ───┤            │            │
                              └──(no)─────────────────────▶ retrieve ──┘            │            │
                                     fix ◀───────────────────────────(fail)─────────┼────────────┘
                                      └──▶ migrate                                  │
                                            ▲──────(re-generate, low confidence)────┘
                                                                                          (pass/done)
                                                                                               ▼
                                     END ◀── observe ◀── service_validate ◀────────────────────┘

``dispatch`` (DispatcherAgent) writes a route plan; ``orchestrate``
(OrchestratorAgent) decomposes the migration into sub-tasks and marks the
independent ones. When it finds concurrent work, ``parallel`` runs those as
compiled subgraphs over isolated state copies and merges the branches (fan-out /
fan-in); otherwise the run takes the original sequential path, routed by the same
``dispatch_condition`` as before. Both paths converge on ``plan``.

``reflect`` (ReflectorAgent) self-critiques the migrated code and routes
low-confidence output back to ``migrate`` with feedback (bounded by
max_reflections); ``service_validate`` (RuntimeValidatorAgent) runs the optional
external validator once the offline validate/fix loop settles; ``observe`` records
terminal metrics.
"""

import logging

from langgraph.graph import END, StateGraph
from memory import build_checkpointer

from .conditions import (
    migrate_condition,
    orchestrate_condition,
    reflect_condition,
    validate_condition,
)
from .nodes import (
    analyze_node,
    deep_analyze_node,
    dispatch_node,
    fix_node,
    migrate_node,
    observe_node,
    orchestrate_node,
    parallel_node,
    plan_node,
    reflect_node,
    retrieve_node,
    service_validate_node,
    validate_node,
)
from .state import GraphState

log = logging.getLogger("CodeMigrateAI.MigrationGraph")


def build_migration_graph(checkpointer=None, settings=None):
    """Construct and compile the migration workflow graph.

    ``checkpointer`` controls graph-state persistence per ``thread_id``:

    ``None`` (default)
        No checkpointer. A compiled graph that *has* one rejects any invocation
        without ``config={"configurable": {"thread_id": ...}}``, so making
        persistence the default would break every direct caller. It also makes
        thread ids load-bearing: two runs sharing an id resume each other's
        state rather than starting clean. Opting in keeps that contract with
        the callers equipped to honour it.
    ``True``
        Build one from settings via :func:`memory.build_checkpointer`
        (AsyncSqliteSaver under a running loop, MemorySaver otherwise).
    a saver instance
        Use it as given.

    :class:`pipeline.orchestrator.Pipeline` opts in and supplies each run's
    session id as the thread id.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("orchestrate", orchestrate_node)
    workflow.add_node("parallel", parallel_node)
    workflow.add_node("deep_analyze", deep_analyze_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("migrate", migrate_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fix", fix_node)
    workflow.add_node("service_validate", service_validate_node)
    workflow.add_node("observe", observe_node)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Analysis -> the DispatcherAgent (writes the route plan) -> the
    # OrchestratorAgent (decomposes the work into sub-tasks). The branch below
    # consults that decomposition: independent sub-tasks go to the `parallel`
    # fan-out node, and anything else falls back to the original sequential path
    # (deep analysis for complex code, otherwise straight to retrieval).
    #
    # All three paths converge on `plan`, so the pre-planning phase is the only
    # thing orchestration reshapes — planning onward is untouched, including the
    # migrate -> retrieve re-retrieval loop, which still re-enters `retrieve`
    # directly rather than re-running a whole fan-out for one targeted query.
    workflow.add_edge("analyze", "dispatch")
    workflow.add_edge("dispatch", "orchestrate")
    workflow.add_conditional_edges(
        "orchestrate",
        orchestrate_condition,
        {
            "parallel": "parallel",
            "deep_analyze": "deep_analyze",
            "retrieve": "retrieve",
        },
    )
    workflow.add_edge("parallel", "plan")
    workflow.add_edge("deep_analyze", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "migrate")

    # Conditional: produced code -> reflect (self-critique before validating);
    # ungrounded imports + budget left -> retrieve (adaptive re-retrieval loop);
    # produced nothing -> service_validate. The reflect node self-skips when
    # reflection is disabled, so this stays a straight path to validate by default.
    workflow.add_conditional_edges(
        "migrate",
        migrate_condition,
        {
            "validate": "reflect",
            "retrieve": "retrieve",
            "end": "service_validate",
        },
    )

    # Conditional: reflection passed (or was skipped) -> validate; low confidence
    # with budget left -> migrate (regenerate with the reflection feedback).
    workflow.add_conditional_edges(
        "reflect",
        reflect_condition,
        {"validate": "validate", "re_migrate": "migrate"},
    )

    # Conditional: pass/exhausted -> service_validate, fail -> fix (retry loop)
    workflow.add_conditional_edges(
        "validate",
        validate_condition,
        {"fix": "fix", "end": "service_validate"},
    )
    workflow.add_edge("fix", "migrate")  # Loop back

    # Optional external service validation once the offline loop settles, then the
    # terminal observability node: every end-path funnels through observe so
    # metrics (and open-circuit state) reflect the whole run before returning.
    workflow.add_edge("service_validate", "observe")
    workflow.add_edge("observe", END)

    if checkpointer is True:
        checkpointer = build_checkpointer(settings)
    app = workflow.compile(checkpointer=checkpointer or None)
    node_count = len(getattr(app, "nodes", {}) or {})
    log.info(
        "Migration graph compiled with %d nodes (checkpointer: %s)",
        node_count,
        type(checkpointer).__name__ if checkpointer else "none",
    )
    return app
