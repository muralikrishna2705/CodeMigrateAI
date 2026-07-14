import logging
from typing import Any, Optional

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ProviderAgent")


class ProviderAgent(BaseAgent):
    name = "ProviderAgent"
    requires_llm = False

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self._providers: dict[str, Any] = {}

    def register(self, name: str, instance: Any):
        self._providers[name] = instance
        log.info("Provider registered: %s", name)

    def get(self, name: str) -> Optional[Any]:
        return self._providers.get(name)

    async def run(self, state: MigrationState) -> AgentResult:
        registered = list(self._providers.keys())
        log.info(
            "ProviderAgent: %d providers available: %s", len(registered), registered
        )
        return AgentResult(
            success=True,
            summary=f"{len(registered)} providers registered",
            details={"providers": registered},
        )
