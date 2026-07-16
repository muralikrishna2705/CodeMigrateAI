import asyncio
import logging

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent
from runtime import agent_recovery

log = logging.getLogger("CodeMigrateAI.ObserverAgent")

_metrics: dict = {
    "agents_run": 0,
    "total_duration_ms": 0,
    "success_count": 0,
    "error_count": 0,
    "by_agent": {},
}


class ObserverAgent(BaseAgent):
    name = "ObserverAgent"
    requires_llm = False
    needs = ("migration_memory",)

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self._memory = (config or {}).get("migration_memory")

    async def run(self, state: MigrationState) -> AgentResult:
        global _metrics
        _metrics["agents_run"] = len(state.agents_done)
        _metrics["total_duration_ms"] = sum(r.duration_ms for r in state.reports)
        _metrics["success_count"] = sum(
            1 for r in state.reports if r.status == "success"
        )
        _metrics["error_count"] = len(state.errors)

        for report in state.reports:
            agent_name = report.agent
            if agent_name not in _metrics["by_agent"]:
                _metrics["by_agent"][agent_name] = {"runs": 0, "errors": 0}
            _metrics["by_agent"][agent_name]["runs"] += 1
            if report.status == "error":
                _metrics["by_agent"][agent_name]["errors"] += 1

        remembered = await self._remember(state)

        summary = (
            f"Observed {len(state.agents_done)} agents "
            f"({_metrics['success_count']} ok, {_metrics['error_count']} errors)"
        )
        if remembered:
            summary += " · migration remembered"

        return AgentResult(
            success=True,
            summary=summary,
            details={**self.get_metrics(), "remembered": remembered},
        )

    async def _remember(self, state: MigrationState) -> bool:
        """Record a clean migration into cross-session memory. Returns success.

        The observer is the terminal graph node, so it is the only place that
        sees a run's final outcome — including whether the fix loop settled on
        valid code.

        Only clean runs are recorded. A migration with errors, or one whose code
        failed validation, is exactly the precedent a future run must not copy:
        ``semantic_search`` presents hits as known-good prior art, so storing a
        bad one would launder a failure into grounding.
        """
        if not self._memory or not get_settings().enable_migration_memory:
            return False
        if state.errors or not state.migrated_code:
            return False
        validation = state.validation_result or {}
        if not validation.get("valid", True):
            return False

        try:
            # Chroma writes (and the embedding call behind them) are blocking;
            # keep them off the event loop like every other store access.
            await asyncio.to_thread(
                self._memory.remember,
                source_code=state.source_code,
                source_language=state.source_language,
                source_version=state.source_version,
                target_language=state.target_language,
                target_version=state.target_version,
                migrated_code=state.migrated_code,
                plan_summary=state.inline_plan,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — memory is an optimization
            log.warning("Could not record migration memory: %s", exc)
            return False

    @classmethod
    def get_metrics(cls) -> dict:
        # The circuit breaker is fed by graph.nodes._make_node (per-node); the
        # observer just surfaces its current state alongside the run metrics.
        return {**_metrics, "circuit": agent_recovery.snapshot()}

    @classmethod
    def reset_metrics(cls):
        _metrics.clear()
        _metrics.update(
            {
                "agents_run": 0,
                "total_duration_ms": 0,
                "success_count": 0,
                "error_count": 0,
                "by_agent": {},
            }
        )
