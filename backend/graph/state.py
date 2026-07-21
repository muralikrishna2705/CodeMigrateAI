"""LangGraph state schema for the migration workflow.

``GraphState`` mirrors the fields of :class:`models.state.MigrationState` (so a
``MigrationState`` can be hydrated from / written back to it in the node layer)
and adds a few graph-only fields used by the retry loop and error recovery.

**Reducers and the delta contract.** Nodes return only what they *changed* (see
``graph.nodes.writeback``), and the annotations below say how each change folds
into the accumulated state. That pairing is load-bearing in both directions:

- A reducer without delta returns would re-append the whole history on every
  node, because each node used to return the entire state dict.
- Delta returns without reducers would make two concurrent branches writing the
  same key an ``InvalidUpdateError`` — "Can receive only one value per step".

Three policies, each previously implemented by hand in the now-deleted
``graph.merge``:

``operator.add``
    Run history that only ever grows. A node contributes its own new entries.

``_keep_highest``
    Loop budgets. A branch that consumed a retry must not have it reset by a
    sibling that did not, so the larger count wins. Writers emit the absolute
    value rather than an increment, which makes the fold idempotent — the same
    delta applied twice is not two retries.

``_merge_dicts``
    ``code_metrics`` is the one field genuinely written from two sides:
    ``AnalyzerAgent`` writes the base metrics and ``DeepAnalyzerAgent`` adds a
    ``deep_analysis`` key. Key-wise merging keeps both; last-write-wins would
    silently drop whichever landed first.

Every other field keeps LangGraph's default replace semantics, and that is a
decision rather than an omission. ``migrated_code`` is legitimately rewritten by
the fix loop, and ``retrieval_requests`` is a queue that ``retrieve_node``
*clears* by writing ``[]`` — under ``operator.add`` an empty list is a no-op, so
the queue could never drain and the re-retrieval loop would spin until its
budget ran out.
"""

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


def _keep_highest(current: Optional[int], incoming: Optional[int]) -> int:
    """Fold loop counters by taking the larger value."""
    return max(current or 0, incoming or 0)


def _merge_dicts(current: Optional[dict], incoming: Optional[dict]) -> Optional[dict]:
    """Fold two metric dicts key-wise, preserving a never-analyzed ``None``.

    Downstream code distinguishes "no metrics" from "empty metrics", so a None
    is carried through rather than normalized into ``{}``.
    """
    if incoming is None:
        return current
    if current is None:
        return dict(incoming)
    return {**current, **incoming}


# Field groups the node layer needs to know about to build a correct delta.
# Defined here, beside the annotations they describe, so a field cannot gain a
# reducer without the writeback path learning how to feed it.
ACCUMULATING_FIELDS = ("reports", "errors", "agents_completed", "subgraph_results")
COUNTER_FIELDS = ("retry_count", "reflection_count", "reretrieval_count")


class GraphState(TypedDict, total=False):
    """LangGraph state schema — extends MigrationState fields."""

    # --- Migration inputs (mirrors MigrationState) ---
    # Never written by a node: set once when the graph is invoked. They are
    # absent from every delta, which is the clearest signal the contract holds.
    source_code: str
    source_language: str
    source_version: str
    target_language: str
    target_version: str

    # --- Working data produced by the agents ---
    migration_type: str
    code_metrics: Annotated[Optional[dict], _merge_dicts]
    inline_plan: str
    migrated_code: str
    rag_context: str
    validation_result: Optional[dict]

    reports: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
    agents_completed: Annotated[list[str], operator.add]

    # --- Adaptive-RAG feedback bus ---
    # Deliberately un-reduced: a consumed queue, cleared by writing [].
    retrieval_requests: list[str]
    reretrieval_count: Annotated[int, _keep_highest]
    max_reretrievals: int

    # --- Dynamic routing (DispatcherAgent -> dispatch_condition) ---
    route_plan: dict

    # --- Persistent memory (Dimension 5) ---
    # Seeded by the pipeline from MigrationMemory.recall before the graph runs,
    # so nodes can ground on prior art without each re-querying the store.
    # session_id doubles as the checkpoint thread_id.
    memory_hits: list[dict]
    session_id: str

    # --- Dynamic orchestration (OrchestratorAgent -> orchestrate_condition) ---
    # ``parallel_tasks`` is the sub-task decomposition the orchestrator planned
    # (task names resolved against graph.subgraphs.SUBGRAPH_TASKS); the graph
    # fans those out with Send and each branch appends its own record to
    # ``subgraph_results`` — {task, ok, agents, error} — which the final report
    # reads. Both stay empty on the sequential path.
    parallel_tasks: list[str]
    subgraph_results: Annotated[list[dict], operator.add]
    # Set on a Send payload, never returned by a node: it tells one fanned-out
    # invocation of the `branch` node which sub-task it is. It is per-invocation
    # input rather than shared state, which is why no reducer has to arbitrate
    # between the concurrent branches that each carry a different value.
    branch_task: str

    # --- Agent reflection (ReflectorAgent -> reflect_condition) ---
    reflection_score: float
    reflection_feedback: str
    reflection_recommendation: str
    reflection_count: Annotated[int, _keep_highest]
    max_reflections: int
    # Seeded from settings by the orchestrator; gates the `reflect` node so offline
    # / direct-graph runs never reflect unless a test opts in.
    enable_reflection: bool

    # --- Graph-specific fields ---
    retry_count: Annotated[int, _keep_highest]
    max_retries: int
    best_effort_code: str
    final_result: Optional[str]
    # Seeded from settings by the orchestrator; gates the post-migration
    # service-validation node so offline runs never touch the validator service.
    enable_validation: bool
    # Seeded from settings by the orchestrator; gates the parallel fan-out so a
    # direct-graph run that never seeds it takes the sequential path.
    parallel_enabled: bool
    max_parallel_tasks: int
