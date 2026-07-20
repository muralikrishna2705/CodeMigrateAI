"""LangGraph agentic system for CodeMigrateAI (Phase 3).

Replaces the linear pipeline with a compiled ``StateGraph`` that supports
conditional branching (complexity-based deep analysis) and a validate -> fix ->
re-migrate retry loop.
"""

from .cicd_graph import build_cicd_graph
from .conditions import (
    complexity_condition,
    orchestrate_condition,
    validate_condition,
)
from .merge import merge_results
from .migration_graph import build_migration_graph
from .parallel import BranchResult, run_parallel
from .state import GraphState
from .subgraphs import SUBGRAPH_TASKS, resolve_tasks

__all__ = [
    "GraphState",
    "build_migration_graph",
    "build_cicd_graph",
    "complexity_condition",
    "orchestrate_condition",
    "validate_condition",
    "BranchResult",
    "run_parallel",
    "merge_results",
    "SUBGRAPH_TASKS",
    "resolve_tasks",
]
