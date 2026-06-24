import json
import logging
import re
from typing import Any, AsyncIterator

import httpx
from config import get_settings

log = logging.getLogger("CodeMigrateAI.LLM")


class LLMClient:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.settings.llm_timeout_sec,
                    write=30.0,
                    pool=5.0,
                )
            )
        return self._client

    async def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        payload = self._build_payload(prompt, system_prompt, stream=False)
        log.info(
            "Ollama call: model=%s, prompt=%d chars",
            self.settings.llm_model,
            len(prompt),
        )

        response = await self.client.post(
            f"{self.settings.ollama_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    async def stream_llm(
        self, prompt: str, system_prompt: str = ""
    ) -> AsyncIterator[str]:
        payload = self._build_payload(prompt, system_prompt, stream=True)

        async with self.client.stream(
            "POST",
            f"{self.settings.ollama_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done", False):
                    break

    def _build_payload(
        self, prompt: str, system_prompt: str, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.settings.llm_temperature,
                "num_predict": self.settings.llm_num_predict,
                "num_ctx": self.settings.llm_num_ctx,
                "num_thread": self.settings.llm_num_threads,
                "top_p": self.settings.llm_top_p,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def extract_json(self, raw_text: str) -> dict:
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                pass
        for obj in sorted(re.findall(r"\{[\s\S]*\}", raw_text), key=len, reverse=True):
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                pass
        raise ValueError("No valid JSON in LLM response")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.settings.ollama_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
