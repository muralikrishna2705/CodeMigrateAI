import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

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

        return AgentResult(
            success=True,
            summary=(
                f"Observed {len(state.agents_done)} agents "
                f"({_metrics['success_count']} ok, {_metrics['error_count']} errors)"
            ),
            details=dict(_metrics),
        )

    @classmethod
    def get_metrics(cls) -> dict:
        return dict(_metrics)

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
