import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent
    from llm.client import LLMClient

log = logging.getLogger("CodeMigrateAI.Pipeline")


class AgentRegistry:
    def __init__(self, llm_client: "LLMClient", settings=None):
        self._llm_client = llm_client
        self._settings = settings
        self._agents: dict[str, "BaseAgent"] = {}
        self._discover_agents()

    def _discover_agents(self):
        # Import all agent modules to trigger AgentMeta registration
        import agents.analyzer_agent  # noqa: F401
        import agents.deep_analyzer_agent  # noqa: F401
        import agents.fixer_agent  # noqa: F401
        import agents.migrator_agent  # noqa: F401
        import agents.planner_agent  # noqa: F401
        import agents.retriever_agent  # noqa: F401
        import agents.validator_agent  # noqa: F401
        import runtime.agent_dispatcher  # noqa: F401
        import runtime.agent_observer  # noqa: F401
        import runtime.agent_providers  # noqa: F401
        import runtime.agent_recovery  # noqa: F401
        import runtime.agent_runtime  # noqa: F401
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

    def get_order(self) -> list["BaseAgent"]:
        # Runtime agents first (setup/teardown), then domain agents
        runtime_order = [
            "ProviderAgent",
            "RuntimeAgent",
            "DispatcherAgent",
            "ObserverAgent",
            "RecoveryAgent",
            "RuntimeValidatorAgent",
        ]
        domain_order = [
            "AnalyzerAgent",
            "DeepAnalyzerAgent",
            "RetrieverAgent",
            "PlannerAgent",
            "MigratorAgent",
            "ValidatorAgent",
            "FixerAgent",
        ]
        order = []
        for name in runtime_order + domain_order:
            if name in self._agents:
                order.append(self._agents[name])
        return order

    def register(self, name: str, agent: "BaseAgent"):
        self._agents[name] = agent

    def attach_rag_pipeline(self, rag_pipeline) -> None:
        """Inject the RAG pipeline into the RetrieverAgent after discovery.

        Agents are constructed at registry-build time, which is before the RAG
        pipeline (ChromaDB + embeddings) is initialized in the app lifespan, so
        the dependency is wired in here once it is available.
        """
        agent = self._agents.get("RetrieverAgent")
        if agent is not None:
            agent._rag_pipeline = rag_pipeline
            log.info("RAG pipeline attached to RetrieverAgent")
