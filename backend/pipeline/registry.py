import logging
from typing import TYPE_CHECKING

from models.state import PipelineMode

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from llm.client import LLMClient

log = logging.getLogger("CodeMigrateAI.Pipeline")


PIPELINE_ORDERS: dict[PipelineMode, list[str]] = {
    PipelineMode.FAST: ["AnalyzerAgent", "PlannerAgent", "MigratorAgent"],
    PipelineMode.DEEP: ["AnalyzerAgent", "PlannerAgent", "MigratorAgent"],
    PipelineMode.VALIDATED: [
        "AnalyzerAgent",
        "PlannerAgent",
        "MigratorAgent",
        "ValidatorAgent",
    ],
}


class AgentRegistry:
    def __init__(self, llm_client: "LLMClient"):
        self._llm_client = llm_client
        self._agents: dict[str, "BaseAgent"] = {}
        self._init_agents()

    def _init_agents(self):
        from agents.analyzer import AnalyzerAgent
        from agents.migrator import MigratorAgent
        from agents.planner import PlannerAgent
        from agents.validator import ValidatorAgent

        self._agents = {
            "AnalyzerAgent": AnalyzerAgent(self._llm_client),
            "PlannerAgent": PlannerAgent(self._llm_client),
            "MigratorAgent": MigratorAgent(self._llm_client),
            "ValidatorAgent": ValidatorAgent(self._llm_client),
        }

    def get_agent(self, name: str) -> "BaseAgent":
        return self._agents[name]

    def get_order(self, mode: PipelineMode) -> list["BaseAgent"]:
        names = PIPELINE_ORDERS.get(mode, PIPELINE_ORDERS[PipelineMode.FAST])
        return [self._agents[name] for name in names]

    def register(self, name: str, agent: "BaseAgent"):
        self._agents[name] = agent
