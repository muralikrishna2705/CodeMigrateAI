import logging
from typing import Optional

from cache.keys import generate_key
from cache.manager import CacheManager
from config import get_settings
from models.state import MigrationState
from pipeline.registry import AgentRegistry

log = logging.getLogger("CodeMigrateAI.Pipeline")


class Pipeline:
    def __init__(
        self,
        llm_client,
        cache_manager: Optional[CacheManager] = None,
        settings=None,
    ):
        self.settings = settings or get_settings()
        self.cache = cache_manager
        self.registry = AgentRegistry(llm_client)

    async def run(self, state: MigrationState) -> MigrationState:
        if self.cache and self.settings.cache_enabled:
            cache_key = generate_key(state)
            cached = await self.cache.get(cache_key)
            if cached:
                log.info("Cache hit for %s...", cache_key[:16])
                return cached

        log.info("=" * 60)
        log.info(
            "Pipeline START: %s %s to %s %s (mode=%s)",
            state.source_language,
            state.source_version,
            state.target_language,
            state.target_version,
            state.pipeline_mode.value,
        )
        log.info("Source size: %d chars", len(state.source_code))
        log.info("=" * 60)

        agents = self.registry.get_order(state.pipeline_mode)

        for agent in agents:
            if agent.name == "MigratorAgent" and self.settings.enable_streaming:
                agent.stream_callback = getattr(state, "_stream_callback", None)

            log.info("Running %s...", agent.name)
            state = await agent(state)

            if state.errors and agent.name == "MigratorAgent":
                log.error("MigratorAgent failed - stopping pipeline")
                break

        state.completed_at = __import__("datetime").datetime.utcnow()

        if (
            self.cache
            and self.settings.cache_enabled
            and state.migrated_code
            and not state.errors
        ):
            cache_key = generate_key(state)
            await self.cache.set(cache_key, state)

        log.info(
            "Pipeline END. Agents done: %s | Errors: %d",
            state.agents_done,
            len(state.errors),
        )
        return state


async def run_migration_pipeline(state: MigrationState) -> MigrationState:
    from cache.manager import CacheManager
    from llm.client import LLMClient

    llm = LLMClient()
    cache = CacheManager() if state.pipeline_mode.value != "fast" else None
    pipeline = Pipeline(llm, cache)

    try:
        return await pipeline.run(state)
    finally:
        await llm.close()
