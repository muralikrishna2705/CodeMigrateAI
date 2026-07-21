"""Adapter presenting a LangChain chat model through this project's LLM API.

Every agent already talks to ``call_llm`` / ``stream_llm``, so this keeps those
signatures byte-for-byte while the implementation underneath becomes a
:class:`~langchain_core.language_models.BaseChatModel` built by
:mod:`llm.providers`. That is what lets the provider swap land without touching
a single agent — call sites migrate to native tool calling and structured output
one at a time, in later phases, rather than all at once.

New code that needs tool binding or structured output should reach for
:meth:`LLMClient.chat_model` and use the model directly; ``call_llm`` is the
compatibility surface, not the target API.
"""

import json
import logging
from typing import AsyncIterator

import httpx
from config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage
from llm import providers

log = logging.getLogger("CodeMigrateAI.LLM")


class LLMClient:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._http: httpx.AsyncClient | None = None

    # --- Model access ------------------------------------------------------

    @property
    def fast_model(self) -> str:
        """Model id for lightweight tasks (routing, grading, planning).

        Resolved through the provider defaults, so it is a real model name even
        when ``fast_llm_model`` is unset — which is what agents' ``model=``
        routing argument expects to receive.
        """
        return providers.resolve_model_name("fast", self.settings)

    @property
    def main_model(self) -> str:
        return providers.resolve_model_name("main", self.settings)

    def chat_model(self, role: str = "main", *, json_mode: bool = False):
        """The underlying chat model, for ``bind_tools`` / ``with_structured_output``."""
        return providers.get_chat_model(
            role, json_mode=json_mode, settings=self.settings
        )

    def _role_for(self, model: str | None) -> str:
        """Map an explicit model name back to a role.

        Agents route to the cheap model by passing ``model=<fast id>`` rather
        than a role name. Recovering the role matters because role carries more
        than the id — the fast role also disables the provider's reasoning
        budget, which is the bulk of the latency saving on short calls.
        """
        if model and model == self.fast_model:
            return "fast"
        return "main"

    @staticmethod
    def _messages(prompt: str, system_prompt: str) -> list:
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    # --- Completion --------------------------------------------------------

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> str:
        role = self._role_for(model)
        chat = providers.get_chat_model(
            role,
            json_mode=(fmt == "json"),
            model=model,
            settings=self.settings,
        )
        log.info(
            "LLM call: %s (role=%s), prompt=%d chars%s",
            model or providers.resolve_model_name(role, self.settings),
            role,
            len(prompt),
            ", json" if fmt else "",
        )
        response = await chat.ainvoke(self._messages(prompt, system_prompt))
        return (response.text or "").strip()

    async def stream_llm(
        self,
        prompt: str,
        system_prompt: str = "",
        fmt: str | None = None,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        role = self._role_for(model)
        chat = providers.get_chat_model(
            role,
            json_mode=(fmt == "json"),
            model=model,
            settings=self.settings,
        )
        async for chunk in chat.astream(self._messages(prompt, system_prompt)):
            # ``.text`` flattens structured content parts (Gemini returns a list
            # when reasoning or citations are attached) down to plain text, so
            # the streamer downstream never has to know which shape arrived.
            text = chunk.text
            if text:
                yield text

    # --- Provider health / model provisioning ------------------------------

    @property
    def http(self) -> httpx.AsyncClient:
        """HTTP client for Ollama's admin endpoints (tags / pull).

        Only the local provider has anything to administer; hosted providers are
        reached exclusively through their LangChain integration.
        """
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        return self._http

    async def health_check(self) -> bool:
        """Whether the configured provider looks usable.

        For a hosted provider this is a *configuration* check, not a network
        probe: ``/migrate`` calls this on every request, so a real round trip
        would spend quota and add latency to answer a question the actual model
        call is about to answer anyway. A missing key is the failure this can
        genuinely catch early, and it is the common one.
        """
        if providers.is_hosted(self.settings):
            return bool(self.settings.google_api_key)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.settings.ollama_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def ensure_model(self, model: str) -> bool:
        """Ensure a local model is available, pulling it if missing.

        A no-op returning True for hosted providers — there is nothing to pull.
        Never raises; callers treat False as "unavailable" and degrade.
        """
        if providers.is_hosted(self.settings):
            return True
        try:
            resp = await self.http.get(f"{self.settings.ollama_url}/api/tags")
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
            async with self.http.stream(
                "POST",
                f"{self.settings.ollama_url}/api/pull",
                json={"name": model, "stream": True},
                timeout=httpx.Timeout(600.0),
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
        if self._http:
            await self._http.aclose()
            self._http = None
