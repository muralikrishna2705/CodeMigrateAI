"""DispatcherAgent — dynamic router the graph's edges consult.

Formerly a prelude agent whose routing decision nothing read (the graph routed
itself with a hardcoded ``if complexity == "high"``). It now runs as the
``dispatch`` graph node right after analysis and writes a **route plan** onto the
state; ``graph.conditions.dispatch_condition`` reads that plan instead of
recomputing the branch, so routing is data-driven and swappable.

Routing is rule-based by default (deterministic, offline-friendly). When
``settings.dispatcher_llm_routing`` is enabled and an LLM is available, it asks
the fast model whether the code warrants deep analysis, falling back to the rules
on any failure. That makes the LLM-first path a drop-in replacement for the rule
body without touching the graph.
"""

import logging

from config import get_settings
from models.schemas import RouteDecision
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.DispatcherAgent")


class DispatcherAgent(BaseAgent):
    name = "DispatcherAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        complexity = (state.code_metrics or {}).get("complexity", "low")
        migration_type = state.migration_type.value

        deep_analyze = complexity == "high"
        if get_settings().dispatcher_llm_routing:
            llm_deep = await self._llm_wants_deep_analysis(state)
            if llm_deep is not None:
                deep_analyze = llm_deep

        route_plan = self._build_plan(deep_analyze, complexity, migration_type)
        state.route_plan = route_plan
        log.info(
            "DispatcherAgent: complexity=%s type=%s -> %s",
            complexity,
            migration_type,
            route_plan["sequence"],
        )
        return AgentResult(
            success=True,
            summary=(
                f"Routed {complexity} complexity → {len(route_plan['sequence'])} "
                f"agent(s){' (+deep analysis)' if deep_analyze else ''}"
            ),
            details=route_plan,
        )

    @staticmethod
    def _build_plan(deep_analyze: bool, complexity: str, migration_type: str) -> dict:
        sequence = ["AnalyzerAgent"]
        if deep_analyze:
            sequence.append("DeepAnalyzerAgent")
        sequence += ["RetrieverAgent", "PlannerAgent", "MigratorAgent"]
        return {
            "deep_analyze": deep_analyze,
            "complexity": complexity,
            "migration_type": migration_type,
            "sequence": sequence,
        }

    async def _llm_wants_deep_analysis(self, state: MigrationState) -> bool | None:
        """Ask the fast model whether the migration needs deep analysis.

        Best-effort: returns ``None`` on any failure (stub LLM, model down, an
        unparseable answer) so the caller keeps the rule-based decision.
        """
        prompt = (
            f"A developer is migrating {state.source_language} "
            f"{state.source_version} to {state.target_language} "
            f"{state.target_version}. Detected complexity: "
            f"{(state.code_metrics or {}).get('complexity', 'low')}.\n\n"
            "Should a deep structural analysis pass run before planning? It is "
            "worth the extra latency for generics, reflection, async, deep "
            "inheritance, or heavy standard-library use.\n\nCODE:\n"
            f"{state.source_code[: get_settings().max_llm_code_chars]}"
        )
        # A bool field beats parsing 'yes'/'no': free text left a third outcome
        # (neither prefix matched) that silently fell back to the rule, so a
        # chatty model disabled LLM routing without anything logging that it had.
        decision = await self._call_structured(
            RouteDecision,
            prompt,
            system_prompt="You are triaging a code migration.",
        )
        if decision is None:
            return None
        log.info(
            "LLM routing: deep_analyze=%s (%s)",
            decision.deep_analyze,
            decision.reasoning or "no reason given",
        )
        return decision.deep_analyze
