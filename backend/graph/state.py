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

    # --- Graph-specific fields ---
    retry_count: int
    max_retries: int
    best_effort_code: str
    final_result: Optional[str]
