import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from llm.client import LLMClient

log = logging.getLogger("CodeMigrateAI.Pipeline")

PIPELINE_ORDER = ["AnalyzerAgent", "MigratorAgent"]


class AgentRegistry:
    def __init__(self, llm_client: "LLMClient", settings=None):
        self._llm_client = llm_client
        self._settings = settings
        self._agents: dict[str, "BaseAgent"] = {}
        self._init_agents()

    def _init_agents(self):
        from agents.analyzer import AnalyzerAgent
        from agents.migrator import MigratorAgent

        self._agents = {
            "AnalyzerAgent": AnalyzerAgent(
                self._llm_client,
                {
                    "enable_semantic_analysis": bool(
                        getattr(self._settings, "enable_semantic_analysis", False)
                    )
                },
            ),
            "MigratorAgent": MigratorAgent(self._llm_client),
        }

    def get_agent(self, name: str) -> "BaseAgent":
        return self._agents[name]

    def get_order(self) -> list["BaseAgent"]:
        return [self._agents[name] for name in PIPELINE_ORDER]

    def register(self, name: str, agent: "BaseAgent"):
        self._agents[name] = agent
