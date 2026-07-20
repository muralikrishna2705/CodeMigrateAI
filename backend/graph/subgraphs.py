"""Reusable compiled subgraphs the orchestrator composes a migration from.

Each builder returns an independently compiled ``StateGraph`` covering one phase
of the work. They exist so the orchestrator can treat a phase as a single unit —
invoking it inline, or handing several of them to :func:`graph.parallel.run_parallel`
to run concurrently — instead of the top-level graph hardcoding one fixed edge
order through the individual nodes.

**On the shared state schema.** Every subgraph is compiled against ``GraphState``
rather than a narrower per-subgraph TypedDict. The node functions all hydrate a
full :class:`~models.state.MigrationState` (they need ``source_code``, the
language pair, the accumulated ``reports``), so a reduced schema would have to
re-declare nearly every field and then be translated back at each boundary. What
actually isolates a parallel branch is that it runs against its own deep copy of
the state (see :mod:`graph.parallel`), not a narrower type.

Compiled graphs are cached per builder: compilation walks and validates the whole
graph, and these are rebuilt on every orchestrated migration otherwise.
"""

import logging
from functools import lru_cache

from langgraph.graph import END, StateGraph

from .conditions import validate_condition
from .nodes import (
    deep_analyze_node,
    fix_node,
    migrate_node,
    plan_node,
    reflect_node,
    retrieve_node,
    validate_node,
)
from .state import GraphState

log = logging.getLogger("CodeMigrateAI.Subgraphs")


@lru_cache(maxsize=1)
def build_analysis_subgraph():
    """Deep structural analysis of the source code.

    Depends only on the base metrics ``AnalyzerAgent`` already wrote, which is
    what makes it safe to run alongside retrieval.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("deep_analyze", deep_analyze_node)
    workflow.set_entry_point("deep_analyze")
    workflow.add_edge("deep_analyze", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def build_retrieval_subgraph():
    """RAG context provisioning.

    ``RetrieverAgent`` builds its query from the source code and the base
    metrics' ``key_constructs`` — never from the deep-analysis output — so this
    has no data dependency on the analysis subgraph.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def build_migration_subgraph():
    """Plan then generate the migrated code."""
    workflow = StateGraph(GraphState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("migrate", migrate_node)
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "migrate")
    workflow.add_edge("migrate", END)
    return workflow.compile()


@lru_cache(maxsize=1)
def build_fix_subgraph():
    """Validate, and repair while validation fails and retries remain.

    Carries the ``validate -> fix -> migrate -> validate`` loop, bounded exactly
    as in the top-level graph: ``fix_node`` advances ``retry_count`` and
    ``validate_condition`` stops at ``max_retries``.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("validate", validate_node)
    workflow.add_node("fix", fix_node)
    workflow.add_node("migrate", migrate_node)
    workflow.set_entry_point("validate")
    workflow.add_conditional_edges(
        "validate",
        validate_condition,
        {"fix": "fix", "end": END},
    )
    workflow.add_edge("fix", "migrate")
    workflow.add_edge("migrate", "validate")
    return workflow.compile()


@lru_cache(maxsize=1)
def build_reflection_subgraph():
    """Self-critique of the migrated code.

    A single node: ``reflect_node`` self-skips when reflection is disabled, and
    the regeneration loop it drives lives in the top-level graph (where the
    ``reflect -> migrate`` edge can reach the real migrate node).
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("reflect", reflect_node)
    workflow.set_entry_point("reflect")
    workflow.add_edge("reflect", END)
    return workflow.compile()


# --- Task catalog ---------------------------------------------------------
#
# Maps the sub-task names the OrchestratorAgent emits onto subgraph builders.
# The orchestrator plans against these names and `parallel_node` resolves them
# here, so adding a composable phase is a builder plus an entry — the graph
# wiring does not change.

SUBGRAPH_TASKS: dict[str, callable] = {
    "analysis": build_analysis_subgraph,
    "retrieval": build_retrieval_subgraph,
    "migration": build_migration_subgraph,
    "fix": build_fix_subgraph,
    "reflection": build_reflection_subgraph,
}


def resolve_tasks(names) -> list[tuple[str, object]]:
    """Resolve task names to ``(name, compiled subgraph)`` pairs.

    Unknown names are skipped with a warning rather than raising: the names can
    come from an LLM decomposition, and one hallucinated entry must not abort a
    migration that the remaining tasks can still complete.
    """
    resolved: list[tuple[str, object]] = []
    for name in names or []:
        builder = SUBGRAPH_TASKS.get(name)
        if builder is None:
            log.warning("Unknown subgraph task %r; skipping", name)
            continue
        resolved.append((name, builder()))
    return resolved
