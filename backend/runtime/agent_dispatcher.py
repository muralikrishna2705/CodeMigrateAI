import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.DispatcherAgent")


class DispatcherAgent(BaseAgent):
    name = "DispatcherAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        complexity = (state.code_metrics or {}).get("complexity", "low")
        migration_type = state.migration_type.value

        routing = {
            "low": ["AnalyzerAgent", "PlannerAgent", "MigratorAgent"],
            "medium": ["AnalyzerAgent", "PlannerAgent", "MigratorAgent"],
            "high": [
                "AnalyzerAgent",
                "DeepAnalyzerAgent",
                "PlannerAgent",
                "MigratorAgent",
            ],
        }

        pipeline = routing.get(complexity, routing["low"])
        log.info(
            "DispatcherAgent: complexity=%s migration_type=%s pipeline=%s",
            complexity,
            migration_type,
            pipeline,
        )
        return AgentResult(
            success=True,
            summary=f"Routing: {complexity} complexity → {len(pipeline)} agents",
            details={
                "complexity": complexity,
                "migration_type": migration_type,
                "pipeline": pipeline,
            },
        )
