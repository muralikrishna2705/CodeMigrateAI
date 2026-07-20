"""Fan-in: fold parallel subgraph branches back into one ``GraphState``.

Every branch produced by :func:`graph.parallel.run_parallel` started from a deep
copy of the same inbound state, so each returned state is "the base plus whatever
that branch did". Merging is therefore a **delta** operation, not a concatenation:
appending a branch's ``reports`` wholesale would re-add every report the run had
already accumulated before the fan-out, once per branch.

Conflict policy, applied in task order so the result never depends on which
branch happened to finish first:

- **accumulating lists** (``reports``, ``errors``, ``agents_completed``,
  ``retrieval_requests``) — each branch's delta is appended in task order.
- **``code_metrics``** — a shallow key-wise merge of each branch's added/changed
  keys onto the base. This is the field that actually gets touched from two
  sides: ``AnalyzerAgent`` writes the base metrics before the fan-out and
  ``DeepAnalyzerAgent`` enriches its own copy with a ``deep_analysis`` key, so a
  key-wise merge keeps both instead of one branch's dict overwriting the other's.
- **scalar fields** (``rag_context``, ``inline_plan``, ``migrated_code``, …) —
  the first branch in task order that changed the value wins; a second branch
  changing the same field is logged as a conflict and ignored.
- **counters** (``reretrieval_count``, …) — the maximum across branches, so a
  budget consumed inside any branch stays consumed after the merge.
"""

import logging

from .parallel import BranchResult

log = logging.getLogger("CodeMigrateAI.GraphMerge")

# Lists that grow as the run progresses: merged by appending each branch's delta.
ACCUMULATING_LISTS = (
    "reports",
    "errors",
    "agents_completed",
    "retrieval_requests",
)

# Fields holding a single value produced by at most one branch.
SCALAR_FIELDS = (
    "migration_type",
    "inline_plan",
    "rag_context",
    "migrated_code",
    "validation_result",
    "route_plan",
    "reflection_score",
    "reflection_feedback",
    "reflection_recommendation",
    "best_effort_code",
)

# Loop budgets: a branch that consumed one must not have it reset by a sibling.
COUNTER_FIELDS = (
    "reretrieval_count",
    "reflection_count",
    "retry_count",
)


def merge_results(base: dict, branches: list[BranchResult]) -> dict:
    """Combine ``branches`` into a new state derived from ``base``.

    ``base`` is left unmodified; the merged state is returned. Failed branches
    (``BranchResult.ok`` is False) contribute nothing but still appear in
    ``subgraph_results`` so the failure is visible in the final report.
    """
    merged = dict(base)
    ok_branches = [b for b in branches if b.ok]

    if ok_branches:
        _merge_accumulating_lists(merged, base, ok_branches)
        _merge_code_metrics(merged, base, ok_branches)
        _merge_scalars(merged, base, ok_branches)
        _merge_counters(merged, base, ok_branches)

    base_agents = list(base.get("agents_completed") or [])
    merged["subgraph_results"] = list(base.get("subgraph_results") or []) + [
        b.to_dict(base_agents=base_agents) for b in branches
    ]

    log.info(
        "Merged %d/%d branch(es): agents=%s",
        len(ok_branches),
        len(branches),
        merged.get("agents_completed"),
    )
    return merged


def _merge_accumulating_lists(
    merged: dict, base: dict, branches: list[BranchResult]
) -> None:
    for field in ACCUMULATING_LISTS:
        base_items = list(base.get(field) or [])
        combined = list(base_items)
        for branch in branches:
            branch_items = list((branch.state or {}).get(field) or [])
            # The branch started from a copy of base, so everything past the
            # base-length prefix is this branch's own contribution. Guard the
            # short case: a branch that *replaced* the list with something
            # shorter has no meaningful delta to take.
            if len(branch_items) > len(base_items):
                combined.extend(branch_items[len(base_items) :])
        merged[field] = combined


def _merge_code_metrics(merged: dict, base: dict, branches: list[BranchResult]) -> None:
    base_metrics = base.get("code_metrics")
    combined = dict(base_metrics or {})
    touched = False
    for branch in branches:
        branch_metrics = (branch.state or {}).get("code_metrics") or {}
        for key, value in branch_metrics.items():
            if key not in combined or combined[key] != value:
                combined[key] = value
                touched = True
    # Preserve a None (never-analyzed) rather than substituting an empty dict:
    # downstream code distinguishes "no metrics" from "empty metrics".
    if touched or base_metrics is not None:
        merged["code_metrics"] = combined


def _merge_scalars(merged: dict, base: dict, branches: list[BranchResult]) -> None:
    for field in SCALAR_FIELDS:
        base_value = base.get(field)
        winner = None
        for branch in branches:
            value = (branch.state or {}).get(field)
            if value == base_value or not value:
                continue
            if winner is None:
                merged[field] = value
                winner = branch.task
            else:
                log.warning(
                    "Merge conflict on %r: %r already set it, ignoring %r",
                    field,
                    winner,
                    branch.task,
                )


def _merge_counters(merged: dict, base: dict, branches: list[BranchResult]) -> None:
    for field in COUNTER_FIELDS:
        values = [base.get(field, 0) or 0]
        for branch in branches:
            values.append((branch.state or {}).get(field, 0) or 0)
        merged[field] = max(values)
