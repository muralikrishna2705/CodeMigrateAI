"""
CodeMigrateAI — Backend
MTech Final Year Project

Three-agent pipeline:
  AnalyzerAgent → PlannerAgent → MigratorAgent

LLM: deepseek-coder:1.3b via Ollama (~800 MB, free, open-source)
     Ollama runs on Windows host — Docker reaches it via host.docker.internal
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cache.manager import CacheManager
from config import get_settings
from llm.client import LLMClient
from llm.streaming import sse_event_generator
from models.requests import MigrateRequest, MigrateResponse
from models.state import MigrationState, PipelineMode
from pipeline.orchestrator import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("CodeMigrateAI")

llm_client: LLMClient | None = None
pipeline: Pipeline | None = None
cache_manager: CacheManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, pipeline, cache_manager

    settings = get_settings()
    log.info("OLLAMA_URL : %s", settings.ollama_url)
    log.info("LLM_MODEL  : %s", settings.llm_model)
    log.info("PIPELINE_MODE: %s", settings.default_pipeline_mode)

    llm_client = LLMClient()
    alive = await llm_client.health_check()
    if alive:
        log.info("Ollama is reachable and ready")
    else:
        log.warning("Ollama not reachable — check that Ollama is running on Windows")

    cache_manager = CacheManager()
    pipeline = Pipeline(llm_client, cache_manager)

    yield

    await llm_client.close()
    log.info("Shutdown complete")


app = FastAPI(
    title="CodeMigrateAI",
    description="AI-driven code migration — MTech Final Year Project",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check — used by the frontend status pill."""
    alive = await llm_client.health_check()
    return {
        "status": "ok",
        "model": get_settings().llm_model,
        "ollama": "connected" if alive else "unavailable",
    }


@app.get("/languages")
async def get_languages():
    """All languages shown in the source / target dropdowns."""
    return {"languages": get_settings().supported_languages}


@app.post("/migrate", response_model=MigrateResponse)
async def migrate(request: MigrateRequest):
    """
    Core endpoint. Runs AnalyzerAgent → PlannerAgent → MigratorAgent
    and returns the migrated code. Called by the frontend Run Migration button.
    """
    if not await llm_client.health_check():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {get_settings().ollama_url}. "
                f"Make sure Ollama is running on Windows "
                f"and model '{get_settings().llm_model}' is pulled."
            ),
        )

    state = MigrationState(
        source_code=request.source_code,
        source_language=request.source_language,
        source_version=request.source_version,
        target_language=request.target_language,
        target_version=request.target_version,
        pipeline_mode=PipelineMode(
            request.pipeline_mode or get_settings().default_pipeline_mode
        ),
    )

    try:
        final_state = await pipeline.run(state)
    except Exception as exc:
        log.exception("Pipeline crashed")
        raise HTTPException(status_code=500, detail=str(exc))

    return MigrateResponse(
        success=bool(final_state.migrated_code) and not final_state.errors,
        migrated_code=final_state.migrated_code,
        migration_type=final_state.migration_type.value,
        source_language=final_state.source_language,
        source_version=final_state.source_version,
        target_language=final_state.target_language,
        target_version=final_state.target_version,
        reports=[r.model_dump() for r in final_state.reports],
        errors=final_state.errors,
        agents_completed=final_state.agents_done,
        validation_result=final_state.validation_result,
    )


@app.post("/migrate/stream")
async def migrate_stream(request: MigrateRequest):
    """
    SSE streaming endpoint. Returns Server-Sent Events with:
    - agent_start: when an agent begins
    - agent_complete: when an agent finishes
    - token: incremental code tokens from MigratorAgent
    - complete: final result
    - error: any error
    """
    if not await llm_client.health_check():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {get_settings().ollama_url}. "
                f"Make sure Ollama is running on Windows "
                f"and model '{get_settings().llm_model}' is pulled."
            ),
        )

    state = MigrationState(
        source_code=request.source_code,
        source_language=request.source_language,
        source_version=request.source_version,
        target_language=request.target_language,
        target_version=request.target_version,
        pipeline_mode=PipelineMode(
            request.pipeline_mode or get_settings().default_pipeline_mode
        ),
    )

    # Create queue for SSE events
    queue = asyncio.Queue()
    state._stream_queue = queue

    async def event_generator():
        async for event in sse_event_generator(queue):
            yield event

    # Run pipeline in background
    async def run_pipeline():
        try:
            await pipeline.run(state)
        except Exception as e:
            log.exception("Streaming pipeline error")
            await queue.put({"type": "error", "message": str(e)})

    asyncio.create_task(run_pipeline())

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/cache/stats")
async def cache_stats():
    return cache_manager.stats()


@app.post("/cache/clear")
async def clear_cache():
    cache_manager.clear()
    return {"status": "cleared"}
