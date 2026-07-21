"""OrchestratorAgent — Plan-and-Execute decomposition for the migration graph.

Runs as the ``orchestrate`` graph node right after ``dispatch``. Where the
:class:`~runtime.agent_dispatcher.DispatcherAgent` answers *"which route"*, this
agent answers *"which sub-tasks, and which of them are independent"*: it writes
``parallel_tasks`` onto the state, and ``graph.conditions.orchestrate_condition``
sends the run down the parallel path only when the decomposition actually found
concurrent work. The graph's ``parallel`` node then resolves those names to
compiled subgraphs (``graph.subgraphs.SUBGRAPH_TASKS``) and fans them out.

This is the *plan* half of Plan-and-Execute; the execute half is the fan-out plus
:mod:`graph.merge`. Splitting them keeps the planning swappable — the rule body
below and the optional LLM decomposition produce the same artifact, so neither
the graph wiring nor the executor knows which one ran.

Decomposition is rule-based by default (deterministic, offline-friendly),
mirroring the DispatcherAgent. The rules encode one real data dependency:
``DeepAnalyzerAgent`` and ``RetrieverAgent`` both consume only the base metrics
``AnalyzerAgent`` already wrote and neither reads the other's output, so when
deep analysis is warranted the two are safe to run concurrently. Planning is
strictly advisory — an empty or single-task plan simply routes down the original
sequential flow.
"""

import logging

from config import get_settings
from models.schemas import SubTaskPlan
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.OrchestratorAgent")

# Sub-tasks this agent is allowed to schedule concurrently. `migration`, `fix`
# and `reflection` are deliberately excluded: they form a strict chain (each
# consumes the previous one's `migrated_code`), so they stay on the main graph
# where their loops can run. Validated against graph.subgraphs.SUBGRAPH_TASKS.
PARALLELIZABLE_TASKS = ("analysis", "retrieval")


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        route_plan = state.route_plan or {}
        deep_analyze = bool(route_plan.get("deep_analyze"))

        tasks = self._decompose(deep_analyze)
        if get_settings().orchestrator_llm_planning:
            llm_tasks = await self._llm_decompose(state, deep_analyze)
            if llm_tasks is not None:
                tasks = llm_tasks

        # `parallelizable` is a fact about the *decomposition* — that these
        # sub-tasks have no data dependency on each other — not a promise that
        # the run will fan out. Whether it actually does is decided downstream by
        # `orchestrate_condition` against the `parallel_enabled` kill switch, and
        # what actually happened is recorded in `subgraph_results`. Keeping the
        # two separate is what lets the switch be flipped off without making this
        # agent's report claim something the run didn't do.
        parallelizable = len(tasks) > 1
        state.parallel_tasks = tasks if parallelizable else []

        # Record the decomposition alongside the routing decision so the whole
        # plan for the run is readable from one place in the final report.
        merged_plan = dict(route_plan)
        merged_plan["sub_tasks"] = tasks
        merged_plan["parallelizable"] = parallelizable
        state.route_plan = merged_plan

        log.info(
            "Orchestrator decomposed into %s (%s)",
            tasks,
            "independent" if parallelizable else "single task",
        )
        return AgentResult(
            success=True,
            summary=(
                f"Decomposed into {len(tasks)} sub-task(s): {', '.join(tasks)}"
                f"{' (independent)' if parallelizable else ''}"
            ),
            details={"sub_tasks": tasks, "parallelizable": parallelizable},
        )

    @staticmethod
    def _decompose(deep_analyze: bool) -> list[str]:
        """Rule-based decomposition.

        Only deep analysis unlocks concurrency: without it there is a single
        pre-planning task (retrieval) and nothing to overlap it with, so the run
        takes the sequential path and skips the fan-out machinery entirely.
        """
        if deep_analyze:
            return ["analysis", "retrieval"]
        return ["retrieval"]

    async def _llm_decompose(
        self, state: MigrationState, deep_analyze: bool
    ) -> list[str] | None:
        """Ask the fast model which sub-tasks this migration needs.

        Best-effort: returns ``None`` on any failure (no LLM, model down,
        unparseable answer, nothing valid selected) so the caller keeps the
        rule-based plan. The response is filtered against
        :data:`PARALLELIZABLE_TASKS` — a decomposition naming ``migration`` or a
        hallucinated task would otherwise schedule chained work concurrently.
        """
        prompt = (
            f"A developer is migrating {state.source_language} "
            f"{state.source_version} to {state.target_language} "
            f"{state.target_version}.\n\n"
            "Which preparation sub-tasks should run before planning the "
            "migration?\n\n"
            f"Detected complexity: "
            f"{(state.code_metrics or {}).get('complexity', 'low')}."
        )
        plan = await self._call_structured(
            SubTaskPlan,
            prompt,
            system_prompt="You are planning the preparation phase of a migration.",
        )
        if plan is None:
            return None

        # Preserve the catalog's order rather than the model's, and drop
        # duplicates, so the plan is a stable set regardless of how the model
        # phrased it. The schema's Literal already excludes a hallucinated task
        # name, so this is now ordering and dedup only — not validation.
        tasks = [t for t in PARALLELIZABLE_TASKS if t in plan.tasks]
        if not tasks:
            log.info("LLM decomposition selected nothing valid; using rules")
            return None
        if tasks != self._decompose(deep_analyze):
            log.info("LLM decomposition overrode rules: %s", tasks)
        return tasks
