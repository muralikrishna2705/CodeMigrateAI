import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from models.state import MigrationState

log = logging.getLogger("CodeMigrateAI.Agents")


@dataclass
class AgentResult:
    success: bool
    summary: str
    details: dict | None = None
    error: str | None = None


class BaseAgent(ABC):
    name: str = "BaseAgent"
    requires_llm: bool = True

    def __init__(self, llm_client, config: dict | None = None):
        self.llm = llm_client
        self.config = config or {}

    @abstractmethod
    async def run(self, state: MigrationState) -> AgentResult:
        pass

    async def __call__(self, state: MigrationState) -> MigrationState:
        start = time.perf_counter()
        log.info("[%s] Starting", self.name)

        if not self.should_run(state):
            state.record_skip(self.name, "Skipped by agent configuration")
            return state

        try:
            result = await self.run(state)
            duration_ms = int((time.perf_counter() - start) * 1000)

            if result.success:
                state.record_success(
                    self.name, result.summary, result.details, duration_ms
                )
                log.info(
                    "[%s] Completed in %dms: %s", self.name, duration_ms, result.summary
                )
            else:
                state.record_error(
                    self.name, result.error or result.summary, duration_ms
                )
                log.error(
                    "[%s] Failed in %dms: %s", self.name, duration_ms, result.error
                )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            log.exception("[%s] Exception after %dms", self.name, duration_ms)
            state.record_error(self.name, str(e), duration_ms)

        return state

    def should_run(self, state: MigrationState) -> bool:
        return True
