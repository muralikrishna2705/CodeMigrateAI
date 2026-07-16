"""Build and compile the migration StateGraph.

Flow::

    analyze ──▶ dispatch ──(deep?)──▶ deep_analyze ──▶ retrieve ──▶ plan ──▶ migrate ──▶ validate
                    └──────(no)────────────────────▶ retrieve                     │           │
                                       fix ◀─────────────────────────────────────(fail)───────┘
                                        └──▶ migrate                                           │
                                                                                          (pass/done)
                                                                                               ▼
                                     END ◀── observe ◀── service_validate ◀────────────────────┘

``dispatch`` (DispatcherAgent) writes a route plan the branch consults; ``retrieve``
(RetrieverAgent) provisions RAG context; ``service_validate`` (RuntimeValidatorAgent)
runs the optional external validator once the offline validate/fix loop settles;
``observe`` records terminal metrics.
"""

import logging

from langgraph.graph import END, StateGraph

from .conditions import dispatch_condition, migrate_condition, validate_condition
from .nodes import (
    analyze_node,
    deep_analyze_node,
    dispatch_node,
    fix_node,
    migrate_node,
    observe_node,
    plan_node,
    retrieve_node,
    service_validate_node,
    validate_node,
)
from .state import GraphState

log = logging.getLogger("CodeMigrateAI.MigrationGraph")


def build_migration_graph():
    """Construct and compile the migration workflow graph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("deep_analyze", deep_analyze_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("migrate", migrate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fix", fix_node)
    workflow.add_node("service_validate", service_validate_node)
    workflow.add_node("observe", observe_node)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Analysis -> the DispatcherAgent, which writes the route plan the branch below
    # consults (deep analysis for complex code, otherwise straight to retrieval).
    # Both paths converge on retrieve, which provisions RAG context before planning.
    workflow.add_edge("analyze", "dispatch")
    workflow.add_conditional_edges(
        "dispatch",
        dispatch_condition,
        {"deep_analyze": "deep_analyze", "retrieve": "retrieve"},
    )
    workflow.add_edge("deep_analyze", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "migrate")

    # Conditional: produced code -> validate; ungrounded imports + budget left ->
    # retrieve (adaptive re-retrieval loop); produced nothing -> service_validate.
    workflow.add_conditional_edges(
        "migrate",
        migrate_condition,
        {
            "validate": "validate",
            "retrieve": "retrieve",
            "end": "service_validate",
        },
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

    app = workflow.compile()
    node_count = len(getattr(app, "nodes", {}) or {})
    log.info("Migration graph compiled with %d nodes", node_count)
    return app
