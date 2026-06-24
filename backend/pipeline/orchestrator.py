import logging
from typing import Optional

from cache.keys import generate_key
from cache.manager import CacheManager
from config import get_settings
from llm.streaming import SSEStreamHandler
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
        self.registry = AgentRegistry(llm_client, self.settings)

    async def run(self, state: MigrationState) -> MigrationState:
        if self.cache and self.settings.cache_enabled:
            cache_key = generate_key(state)
            cached = await self.cache.get(cache_key)
            if cached:
                log.info("Cache hit for %s...", cache_key[:16])
                await self._run_optional_validation(cached)
                return cached

        log.info("=" * 60)
        log.info(
            "Pipeline START: %s %s to %s %s",
            state.source_language,
            state.source_version,
            state.target_language,
            state.target_version,
        )
        log.info("Source size: %d chars", len(state.source_code))
        log.info("=" * 60)

        agents = self.registry.get_order()

        # Create streaming handler if enabled
        stream_handler = None
        if self.settings.enable_streaming and hasattr(state, "_stream_queue"):
            stream_handler = SSEStreamHandler(state._stream_queue)

        for agent in agents:
            # Send agent start event
            if stream_handler:
                await stream_handler.send_agent_start(
                    agent.name, f"Starting {agent.name}..."
                )

            # Set stream callback for MigratorAgent
            if (
                agent.name == "MigratorAgent"
                and self.settings.enable_streaming
                and stream_handler
            ):
                agent.stream_callback = stream_handler.send_token

            log.info("Running %s...", agent.name)
            state = await agent(state)

            # Send agent complete event
            if stream_handler:
                # Get the last report for this agent
                agent_report = next(
                    (r for r in reversed(state.reports) if r.agent == agent.name),
                    None,
                )
                if agent_report:
                    await stream_handler.send_agent_complete(
                        agent.name, agent_report.model_dump()
                    )

            if state.errors and agent.name == "MigratorAgent":
                log.error("MigratorAgent failed - stopping pipeline")
                if stream_handler:
                    await stream_handler.send_error(
                        f"MigratorAgent failed: {state.errors[-1]}"
                    )
                break

        await self._run_optional_validation(state)

        state.completed_at = __import__("datetime").datetime.utcnow()

        if (
            self.cache
            and self.settings.cache_enabled
            and state.migrated_code
            and not state.errors
        ):
            cache_key = generate_key(state)
            await self.cache.set(cache_key, state)

        if stream_handler:
            await stream_handler.send_complete(state)

        log.info(
            "Pipeline END. Agents done: %s | Errors: %d",
            state.agents_done,
            len(state.errors),
        )
        return state

    async def _run_optional_validation(self, state: MigrationState):
        if (
            not self.settings.enable_validation
            or not state.migrated_code
            or state.errors
            or state.validation_result
        ):
            return

        from clients.validator_client import ValidatorClient

        validator = ValidatorClient(
            base_url=self.settings.validator_url,
            timeout_sec=self.settings.validator_timeout_sec,
        )
        try:
            state.validation_result = await validator.validate(
                code=state.migrated_code,
                language=state.target_language,
                version=state.target_version,
            )
        except Exception as exc:
            log.warning("Validator service unavailable: %s", exc)
            state.validation_result = {
                "valid": False,
                "errors": [],
                "warnings": [
                    {
                        "line": 0,
                        "column": 0,
                        "message": f"Validator service unavailable: {exc}",
                        "severity": "warning",
                    }
                ],
                "service_error": str(exc),
            }
        finally:
            await validator.close()


async def run_migration_pipeline(state: MigrationState) -> MigrationState:
    from cache.manager import CacheManager
    from llm.client import LLMClient

    llm = LLMClient()
    cache = CacheManager()
    pipeline = Pipeline(llm, cache)

    try:
        return await pipeline.run(state)
    finally:
        await llm.close()
