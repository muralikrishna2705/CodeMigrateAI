import asyncio
import json
import logging
from typing import Any

log = logging.getLogger("CodeMigrateAI.Streaming")


class SSEStreamHandler:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def send_token(self, token: str):
        await self.queue.put({"type": "token", "content": token})

    async def send_agent_start(self, agent: str, message: str):
        await self.queue.put(
            {"type": "agent_start", "agent": agent, "message": message}
        )

    async def send_agent_complete(self, agent: str, result: dict):
        await self.queue.put(
            {"type": "agent_complete", "agent": agent, "result": result}
        )

    async def send_complete(self, state: Any):
        result = {
            "success": bool(state.migrated_code) and not state.errors,
            "migrated_code": state.migrated_code,
            "inline_plan": state.inline_plan,
            "migration_type": state.migration_type.value,
            "source_language": state.source_language,
            "source_version": state.source_version,
            "target_language": state.target_language,
            "target_version": state.target_version,
            "reports": [report.model_dump() for report in state.reports],
            "errors": state.errors,
            "agents_completed": state.agents_done,
            "validation_result": state.validation_result,
        }
        await self.queue.put({"type": "complete", "result": result})

    async def send_error(self, message: str):
        await self.queue.put({"type": "error", "message": message})


async def sse_event_generator(queue: asyncio.Queue):
    try:
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in ("complete", "error"):
                break
    except asyncio.CancelledError:
        log.info("SSE stream cancelled")
        raise
    except Exception as e:
        log.exception("SSE generator error")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
