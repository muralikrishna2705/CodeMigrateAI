"""Tool layer primitives: the AgentTool contract, its result type, and the registry.

Agents call tools *actively* — they decide when a tool runs and with what
arguments — rather than receiving pre-computed state from an upstream pass. This
module is the contract that makes that possible.

Why a plain ABC and not ``langchain_core.tools.BaseTool``: wrapping BaseTool only
pays off when something calls ``bind_tools`` on the model so the LLM can emit
native tool calls. This project talks to Ollama's ``/api/generate``, which has no
``tools`` parameter, so nothing can bind them. The BaseTool wrapper would be
inert ceremony. ``ToolRegistry.describe`` instead renders a prompt-facing catalog
for the prompt-driven selection path (see ``BaseAgent._select_tool``), which is
the only tool-selection mechanism this LLM stack can actually support.

Tools never raise. Every call returns a :class:`ToolResult`, with failures
carried in ``error``, because tool calls are best-effort enrichment: a tool that
is down must degrade the result, never fail the migration.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("CodeMigrateAI.Tools")


@dataclass(frozen=True)
class ToolResult:
    """The uniform return of every tool call.

    ``data`` is the tool-specific payload; ``summary`` is a short human-readable
    line that agents fold into their AgentResult summaries and reports.
    """

    tool: str
    success: bool
    data: Any = None
    summary: str = ""
    error: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        """Report-friendly view (used in AgentResult.details, so it must be JSON-safe)."""
        return {
            "tool": self.tool,
            "success": self.success,
            "summary": self.summary,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolCall:
    """A record of one tool invocation, for reports and the tool-loop transcript."""

    tool: str
    arguments: dict = field(default_factory=dict)
    result: ToolResult | None = None


class AgentTool(ABC):
    """Base class for a callable an agent invokes on demand.

    Subclasses implement :meth:`run` and declare ``name``/``description``/
    ``parameters``. Calling the tool goes through ``__call__``, which adds
    timing, the timeout, and error capture so no subclass repeats them.
    """

    name: str = "tool"
    description: str = ""
    # JSON-schema-shaped argument spec. Not fed to a native tool-calling API
    # (Ollama's generate endpoint has none) — it renders into the selection
    # prompt and documents the call signature for humans.
    parameters: dict[str, str] = {}

    #: Per-call ceiling. A hung tool must not hang the migration.
    timeout_sec: float = 20.0

    def __init__(self, timeout_sec: float | None = None) -> None:
        if timeout_sec is not None:
            self.timeout_sec = timeout_sec

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Do the work. Implementations may raise; ``__call__`` catches."""

    async def __call__(self, **kwargs) -> ToolResult:
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.run(**kwargs), timeout=self.timeout_sec
            )
        except asyncio.TimeoutError:
            duration_ms = int((time.perf_counter() - start) * 1000)
            log.warning("[%s] timed out after %.1fs", self.name, self.timeout_sec)
            return ToolResult(
                tool=self.name,
                success=False,
                error=f"Tool timed out after {self.timeout_sec}s",
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001 — tools degrade, never propagate
            duration_ms = int((time.perf_counter() - start) * 1000)
            log.warning("[%s] failed: %s", self.name, exc)
            return ToolResult(
                tool=self.name,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

        duration_ms = int((time.perf_counter() - start) * 1000)
        log.info("[%s] %dms: %s", self.name, duration_ms, result.summary)
        # Subclasses build their ToolResult without timing themselves.
        return ToolResult(
            tool=result.tool or self.name,
            success=result.success,
            data=result.data,
            summary=result.summary,
            error=result.error,
            duration_ms=duration_ms,
        )

    def spec(self) -> dict:
        """The tool's prompt-facing / introspection spec."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


class ToolRegistry:
    """Name -> tool lookup, O(1) via a plain dict.

    Dict-like on purpose (``get``/``[]``/``in``/``len``/iteration) so agents can
    treat ``self.tools`` as the ``dict[str, AgentTool]`` it is conceptually,
    while the class still owns catalog rendering for the selection prompt.
    """

    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            log.debug("Tool %s re-registered", tool.name)
        self._tools[tool.name] = tool
        log.info("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str, default: Any = None) -> AgentTool | None:
        return self._tools.get(name, default)

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names) -> "ToolRegistry":
        """A registry holding only ``names`` that are present.

        Lets each agent be handed the tools relevant to its job instead of the
        whole catalog — which keeps the selection prompt short, and a short
        catalog is what a small model needs to pick correctly at all.
        """
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def describe(self) -> str:
        """Render the catalog for a tool-selection prompt."""
        lines = []
        for tool in self._tools.values():
            args = ", ".join(
                f"{key}: {desc}" for key, desc in (tool.parameters or {}).items()
            )
            lines.append(f"- {tool.name}({args}) — {tool.description}")
        return "\n".join(lines)

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def __getitem__(self, name: str) -> AgentTool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return bool(self._tools)
