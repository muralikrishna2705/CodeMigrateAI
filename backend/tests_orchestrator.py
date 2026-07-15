"""Offline end-to-end tests for Pipeline.run() with the LangGraph backend.

Exercises the full path main.py actually calls: runtime scaffolding agents ->
RetrieverAgent -> compiled migration graph (analyze -> plan -> migrate ->
validate, with the fix retry loop) -> optional external validation -> cache.

Run: pytest tests_orchestrator.py -v
"""

import asyncio
import json

import pytest

from config import Settings
from models.state import MigrationState, MigrationType
from pipeline.orchestrator import Pipeline

VALID_PY = "def greet():\n    return 'hello'\n"


class StubLLM:
    def __init__(self):
        self.calls: list[str] = []

    async def call_llm(
        self, prompt: str, system_prompt: str = "", fmt: str | None = None
    ) -> str:
        self.calls.append(prompt)
        if "MIGRATION PLANNING TASK" in prompt:
            return json.dumps(
                {"plan_summary": "Upgrade the code.", "steps": [], "risk_areas": []}
            )
        if "failed validation" in prompt:
            return VALID_PY
        return json.dumps({"plan_summary": "Migrated.", "migrated_code": VALID_PY})

    async def stream_llm(
        self, prompt: str, system_prompt: str = "", fmt: str | None = None
    ):
        # MigratorAgent switches to token streaming when a stream_callback is
        # wired (see graph/nodes.py:set_stream_callback), same as LLMClient.
        yield await self.call_llm(prompt, system_prompt, fmt=fmt)

    def extract_json(self, raw_text: str) -> dict:
        return json.loads(raw_text)


def _make_settings():
    # A fresh Settings instance (not the lru_cache'd get_settings() singleton)
    # so these overrides don't leak into other tests running in this process.
    return Settings(cache_enabled=False, enable_validation=False)


@pytest.mark.asyncio
async def test_pipeline_run_uses_graph_backend_end_to_end():
    settings = _make_settings()
    pipeline = Pipeline(StubLLM(), cache_manager=None, settings=settings)

    state = MigrationState(
        source_code="print('hello')",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
    )

    result = await pipeline.run(state)

    assert result.migrated_code.strip() == VALID_PY.strip()
    assert result.migration_type == MigrationType.UPGRADE_VERSION
    assert result.completed_at is not None
    # Runtime scaffolding + retriever ran before the graph.
    for name in ("ProviderAgent", "RuntimeAgent", "RetrieverAgent"):
        assert name in result.agents_done
    # Domain flow ran inside the graph.
    for name in ("AnalyzerAgent", "PlannerAgent", "MigratorAgent", "ValidatorAgent"):
        assert name in result.agents_done
    assert result.validation_result["valid"] is True


@pytest.mark.asyncio
async def test_pipeline_run_streams_agent_events():
    settings = _make_settings()
    pipeline = Pipeline(StubLLM(), cache_manager=None, settings=settings)

    state = MigrationState(
        source_code="print('hello')",
        source_language="python",
        source_version="2.7",
        target_language="python",
        target_version="3.12",
    )
    state._stream_queue = asyncio.Queue()

    result = await pipeline.run(state)

    events = []
    while not state._stream_queue.empty():
        events.append(state._stream_queue.get_nowait())

    types = [e["type"] for e in events]
    assert "complete" in types
    assert any(e["type"] == "agent_complete" and e["agent"] == "MigratorAgent" for e in events)
    assert result.migrated_code.strip() == VALID_PY.strip()


async def _main() -> None:
    await test_pipeline_run_uses_graph_backend_end_to_end()
    await test_pipeline_run_streams_agent_events()
    print("All orchestrator verification checks passed.")


if __name__ == "__main__":
    asyncio.run(_main())
