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

    @property
    def fast_model(self) -> str:
        """Model for lightweight tasks (analysis/planning).

        Falls back to the main model when ``fast_llm_model`` is unset, so routing
        is an opt-in optimization that never introduces a second model unless the
        operator configures (and pulls) one.
        """
        return self.settings.fast_llm_model or self.settings.llm_model

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> str:
        payload = self._build_payload(
            prompt, system_prompt, stream=False, fmt=fmt, model=model
        )
        log.info(
            "Ollama call: model=%s, prompt=%d chars%s",
            payload["model"],
            len(prompt),
            f", format={fmt}" if fmt else "",
        )

        response = await self.client.post(
            f"{self.settings.ollama_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    async def stream_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(
            prompt, system_prompt, stream=True, fmt=fmt, model=model
        )

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
        self,
        prompt: str,
        system_prompt: str,
        stream: bool,
        fmt: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.settings.llm_model,
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
        # Ollama's grammar-constrained decoding: when fmt="json" the model is
        # forced to emit a single syntactically valid JSON object (with proper
        # string escaping), which small models like deepseek-coder:1.3b cannot
        # reliably do on their own. The prompt must still ask for JSON.
        if fmt:
            payload["format"] = fmt
        return payload

    def extract_json(self, raw_text: str) -> dict:
        text = raw_text.strip()

        # 1. Strip common preamble/suffix before any parsing.
        cleaned = re.sub(
            r"(?i)^(?:here(?:'s| is) (?:the |my |your )?"
            r"(?:migrated|converted|upgraded) code[:\s]*|output[:\s]*|"
            r"result[:\s]*|sure[^.]*\.)",
            "",
            text,
        ).strip()

        # 2. Extract from markdown fenced blocks (handle both ```json and ```).
        for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", cleaned):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                pass

        # 3. Balanced-brace extraction — find all {…} pairs using depth tracking.
        candidates = []
        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        candidates.append(cleaned[start : i + 1])
                        start = -1
        for candidate in sorted(candidates, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 4. Fallback: try to repair truncated JSON (add missing closing brace).
        for candidate in candidates:
            for fix in [candidate + "}", candidate + '"}}']:
                try:
                    return json.loads(fix)
                except json.JSONDecodeError:
                    pass

        # 5. Last resort: greedy regex (original approach).
        for obj in sorted(
            re.findall(r"\{[\s\S]*\}", text), key=len, reverse=True
        ):
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

    async def ensure_model(self, model: str) -> bool:
        """Ensure an Ollama model is available locally, pulling it if missing.

        Returns True if the model is present (or was successfully pulled),
        False otherwise. Never raises — callers treat False as "unavailable"
        and degrade gracefully.
        """
        try:
            resp = await self.client.get(f"{self.settings.ollama_url}/api/tags")
            resp.raise_for_status()
            installed = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception as exc:
            log.warning("Could not query Ollama models for '%s': %s", model, exc)
            return False

        # Ollama reports names as "name:tag" (e.g. "nomic-embed-text:latest").
        # Match whether the caller passed a bare name or an explicit tag.
        if model in installed or any(
            name.split(":")[0] == model.split(":")[0] for name in installed
        ):
            return True

        log.info("Model '%s' not found locally; pulling from Ollama registry…", model)
        try:
            async with self.client.stream(
                "POST",
                f"{self.settings.ollama_url}/api/pull",
                json={"name": model, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("error"):
                        log.error("Pull failed for '%s': %s", model, data["error"])
                        return False
                    if data.get("status") == "success":
                        log.info("Model '%s' pulled successfully", model)
                        return True
            # Stream ended without an explicit "success" — assume complete.
            return True
        except Exception as exc:
            log.warning("Failed to pull model '%s': %s", model, exc)
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
