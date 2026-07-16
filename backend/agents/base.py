import json
import logging
import time
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from models.state import MigrationState

if TYPE_CHECKING:
    from agents.tools.base import ToolRegistry, ToolResult

log = logging.getLogger("CodeMigrateAI.Agents")


@dataclass
class AgentResult:
    success: bool
    summary: str
    details: dict | None = None
    error: str | None = None


class AgentMeta(ABCMeta):
    """Metaclass that auto-registers all BaseAgent subclasses.

    Extends ABCMeta (not plain ``type``) so it stays compatible with ABC,
    which BaseAgent inherits — otherwise Python raises a metaclass conflict.
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if not hasattr(cls, "_registry"):
            cls._registry = {}
        if name != "BaseAgent" and ABC not in bases:
            cls._registry[name] = cls

    def get_registry(cls) -> dict[str, type]:
        return dict(cls._registry)


class BaseAgent(ABC, metaclass=AgentMeta):
    _registry: dict[str, type] = {}

    name: str = "BaseAgent"
    requires_llm: bool = True
    # Names of shared dependencies this agent needs from the Provider (DI). The
    # graph runtime resolves these and passes them in ``config`` at construction,
    # so agents don't reach for module globals. Empty means "no extra deps".
    needs: tuple[str, ...] = ()

    def __init__(self, llm_client, config: dict | None = None):
        self.llm = llm_client
        self.config = config or {}
        # Tools arrive through the same needs-based DI as every other dependency
        # (`needs = ("tools",)` -> Provider.get_tools()). Agents that declare no
        # need for tools — and unit tests constructing agents bare — get an empty
        # registry rather than None, so `"x" in self.tools` and `_call_tool` work
        # unconditionally and no call site needs a None check.
        tools = self.config.get("tools")
        if tools is None:
            from agents.tools.base import ToolRegistry

            tools = ToolRegistry()
        self.tools: "ToolRegistry" = tools
        self._tool_calls: list[dict] = []

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

    # --- Tool use ---------------------------------------------------------

    async def _call_tool(self, name: str, **kwargs) -> "ToolResult":
        """Invoke a tool by name, recording the call for the agent's report.

        Never raises and never returns None: an unregistered or disabled tool
        comes back as a failed ToolResult, so callers branch on
        ``result.success`` alone rather than juggling None checks and
        exceptions. That uniformity is what lets an agent treat every tool as
        optional enrichment.
        """
        from agents.tools.base import ToolResult

        tool = self.tools.get(name)
        if tool is None:
            log.debug("[%s] tool %r not available", self.name, name)
            result = ToolResult(
                tool=name, success=False, error=f"Tool {name!r} is not available"
            )
        else:
            result = await tool(**kwargs)

        self._tool_calls.append(
            {
                # Arguments are logged by key only: values carry whole source
                # files and would bloat every report with duplicated code.
                "arguments": sorted(kwargs),
                **result.to_dict(),
            }
        )
        return result

    def tool_call_log(self) -> list[dict]:
        """Tool calls made during this run, for AgentResult.details."""
        return list(self._tool_calls)

    async def _select_tool(
        self, goal: str, tools: "ToolRegistry | None" = None
    ) -> tuple[str, dict] | None:
        """Ask the LLM which tool to call for ``goal``. Returns ``(name, args)``.

        This is prompt-driven, not native tool calling: the configured Ollama
        endpoint (``/api/generate``) has no ``tools`` parameter, so the catalog
        goes into the prompt and the model replies with JSON. ``fmt="json"``
        engages Ollama's grammar-constrained decoding, which is what makes a
        small model emit parseable output at all.

        Returns ``None`` on anything unexpected — no LLM, malformed JSON, or a
        hallucinated tool name — so every caller must have a non-LLM fallback.
        With a 1.3b model that path is taken often; it is the normal case, not
        an error.
        """
        registry = tools if tools is not None else self.tools
        if not registry:
            return None
        call_llm = getattr(self.llm, "call_llm", None)
        if call_llm is None:
            return None

        prompt = (
            "You have these tools:\n"
            f"{registry.describe()}\n\n"
            f"GOAL: {goal}\n\n"
            "Choose the single most useful tool and its arguments. Respond with "
            'only JSON: {"tool": "<name>", "arguments": {"<arg>": "<value>"}}. '
            'If no tool helps, respond {"tool": null}.'
        )
        try:
            raw = await call_llm(
                prompt,
                system_prompt="Respond ONLY with the JSON object.",
                fmt="json",
                **self._fast_model_kwargs(),
            )
        except Exception as exc:  # noqa: BLE001 — selection is strictly optional
            log.warning("[%s] tool selection call failed: %s", self.name, exc)
            return None

        try:
            data = self._parse_tool_choice(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] tool selection unparseable: %s", self.name, exc)
            return None
        if not data:
            return None

        name = data.get("tool")
        if not name or name not in registry:
            # A name outside the catalog is the classic small-model failure:
            # inventing a plausible tool. Treat it as "no selection".
            if name:
                log.info("[%s] LLM chose unknown tool %r; ignoring", self.name, name)
            return None

        arguments = data.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        return name, arguments

    def _parse_tool_choice(self, raw: str) -> dict | None:
        """Parse the selection response, tolerating fenced//prefixed JSON."""
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            extract_json = getattr(self.llm, "extract_json", None)
            if extract_json is None:
                return None
            data = extract_json(raw)
        return data if isinstance(data, dict) else None

    def _fast_model_kwargs(self) -> dict:
        """Return call kwargs routing to the fast model, when supported.

        Analysis/planning agents don't need the heavyweight coder model. This
        returns ``{"model": <fast>}`` for a real :class:`LLMClient` and ``{}`` for
        LLM stubs (which expose neither ``fast_model`` nor a ``model`` kwarg), so
        routing is safe to apply unconditionally at call sites.
        """
        fast = getattr(self.llm, "fast_model", None)
        return {"model": fast} if fast else {}
