"""Build and compile the migration StateGraph.

Flow::

    analyze ──(complexity)──▶ deep_analyze ──▶ plan ──▶ migrate ──(has code)──▶ validate
        └───────(low/med)──────────────────────▶ plan     └──(no code)──▶ END      │
                                                                ▲              (validate)
                                          fix ◀──────────────────────────────────────┘
                                           └──▶ migrate                              │
                                                                                     ▼
                                                                                    END
"""

import logging

from langgraph.graph import END, StateGraph

from .conditions import complexity_condition, migrate_condition, validate_condition
from .nodes import (
    analyze_node,
    deep_analyze_node,
    fix_node,
    migrate_node,
    plan_node,
    validate_node,
)
from .state import GraphState

log = logging.getLogger("CodeMigrateAI.MigrationGraph")


def build_migration_graph():
    """Construct and compile the migration workflow graph."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("deep_analyze", deep_analyze_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("migrate", migrate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fix", fix_node)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Conditional: high complexity -> deep analysis, otherwise straight to plan
    workflow.add_conditional_edges(
        "analyze",
        complexity_condition,
        {"deep_analyze": "deep_analyze", "retrieve": "plan"},
    )
    workflow.add_edge("deep_analyze", "plan")
    workflow.add_edge("plan", "migrate")

    # Conditional: MigratorAgent produced code -> validate, produced nothing -> END
    workflow.add_conditional_edges(
        "migrate",
        migrate_condition,
        {"validate": "validate", "end": END},
    )

    # Conditional: pass -> END, fail -> fix (retry loop)
    workflow.add_conditional_edges(
        "validate",
        validate_condition,
        {"fix": "fix", "end": END},
    )
    workflow.add_edge("fix", "migrate")  # Loop back

    app = workflow.compile()
    node_count = len(getattr(app, "nodes", {}) or {})
    log.info("Migration graph compiled with %d nodes", node_count)
    return app
