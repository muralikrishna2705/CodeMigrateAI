import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from llm.client import LLMClient

log = logging.getLogger("CodeMigrateAI.Pipeline")


class AgentRegistry:
    """Discovers and constructs the domain agents via AgentMeta auto-registration.

    The migration flow itself runs inside the LangGraph (see graph/nodes.py),
    which resolves agents by name from ``BaseAgent.get_registry()``. This registry
    remains the single place that imports every agent module so that registration
    happens, and it constructs one instance of each as a startup sanity check.
    """

    def __init__(self, llm_client: "LLMClient", settings=None):
        self._llm_client = llm_client
        self._settings = settings
        self._agents: dict[str, "BaseAgent"] = {}
        self._discover_agents()

    def _discover_agents(self):
        # Import all agent modules to trigger AgentMeta registration.
        import agents.analyzer_agent  # noqa: F401
        import agents.deep_analyzer_agent  # noqa: F401
        import agents.fixer_agent  # noqa: F401
        import agents.migrator_agent  # noqa: F401
        import agents.planner_agent  # noqa: F401
        import agents.retriever_agent  # noqa: F401
        import agents.validator_agent  # noqa: F401
        import runtime.agent_dispatcher  # noqa: F401
        import runtime.agent_observer  # noqa: F401
        import runtime.agent_validator  # noqa: F401

        from agents.base import BaseAgent

        # Per-agent construction config; keeps existing behaviour wired even
        # though discovery is now generic.
        agent_configs = {
            "AnalyzerAgent": {
                "enable_semantic_analysis": bool(
                    getattr(self._settings, "enable_semantic_analysis", False)
                )
            },
        }

        for name, cls in BaseAgent.get_registry().items():
            self._agents[name] = cls(self._llm_client, agent_configs.get(name))

        log.info(
            "Discovered %d agents: %s", len(self._agents), list(self._agents.keys())
        )

    def get_agent(self, name: str) -> "BaseAgent":
        return self._agents[name]

    def register(self, name: str, agent: "BaseAgent"):
        self._agents[name] = agent
