"""LangGraph state schema for the migration workflow.

``GraphState`` mirrors the fields of :class:`models.state.MigrationState` (so a
``MigrationState`` can be hydrated from / written back to it in the node layer)
and adds a few graph-only fields used by the retry loop and error recovery.
"""

from typing import Optional

from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """LangGraph state schema — extends MigrationState fields."""

    # --- Migration inputs (mirrors MigrationState) ---
    source_code: str
    source_language: str
    source_version: str
    target_language: str
    target_version: str

    # --- Working data produced by the agents ---
    migration_type: str
    code_metrics: Optional[dict]
    inline_plan: str
    migrated_code: str
    rag_context: str
    validation_result: Optional[dict]

    reports: list[dict]
    errors: list[str]
    agents_completed: list[str]

    # --- Adaptive-RAG feedback bus ---
    retrieval_requests: list[str]
    reretrieval_count: int
    max_reretrievals: int

    # --- Dynamic routing (DispatcherAgent -> dispatch_condition) ---
    route_plan: dict

    # --- Dynamic orchestration (OrchestratorAgent -> orchestrate_condition) ---
    # ``parallel_tasks`` is the sub-task decomposition the orchestrator planned
    # (task names resolved against graph.subgraphs.SUBGRAPH_TASKS); the parallel
    # node fans those out and appends one record per branch to
    # ``subgraph_results`` — {task, ok, agents, error} — which the merge node and
    # the final report read. Both stay empty on the sequential path.
    parallel_tasks: list[str]
    subgraph_results: list[dict]

    # --- Agent reflection (ReflectorAgent -> reflect_condition) ---
    reflection_score: float
    reflection_feedback: str
    reflection_recommendation: str
    reflection_count: int
    max_reflections: int
    # Seeded from settings by the orchestrator; gates the `reflect` node so offline
    # / direct-graph runs never reflect unless a test opts in.
    enable_reflection: bool

    # --- Graph-specific fields ---
    retry_count: int
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
