"""Tool layer primitives: the AgentTool contract, its result type, and the registry.

Agents call tools *actively* — they decide when a tool runs and with what
arguments — rather than receiving pre-computed state from an upstream pass. This
module is the contract that makes that possible.

Tools are usable two ways, from one definition:

**Directly**, by an agent that knows what it wants — ``self._call_tool("vector_db",
query=...)``. This is the deterministic path and it stays exactly as it was.

**Natively**, by the model itself. :meth:`AgentTool.as_langchain_tool` adapts a
tool into a ``langchain_core.tools.StructuredTool`` so it can be bound with
``bind_tools`` and executed by ``langgraph.prebuilt.ToolNode``. The model emits a
real tool call against the declared ``args_schema`` instead of being asked to
describe one in prose.

The adapter approach is deliberate: it keeps the timeout, the error capture, and
the ``ToolResult`` contract in one place, so the native path inherits all three
rather than reimplementing them per ``@tool`` function.

Tools never raise. Every call returns a :class:`ToolResult`, with failures
carried in ``error``, because tool calls are best-effort enrichment: a tool that
is down must degrade the result, never fail the migration. On the native path
that failure is handed back to the model as text, so it can react — retry with a
different query, or choose another tool — rather than aborting the run.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

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

    def for_model(self) -> str:
        """What the model reads back after calling this tool.

        A failure is reported as text rather than raised, so the model can react
        to it — pick a different tool, broaden a query — instead of the run
        aborting. That is the whole reason tools never raise.
        """
        if not self.success:
            return f"ERROR: {self.error or 'tool failed'}"
        parts = [self.summary] if self.summary else []
        payload = self.data
        if isinstance(payload, dict):
            # `context` is pre-rendered prose meant for a prompt; anything else
            # is structured and reads better as compact JSON than as a repr.
            if payload.get("context"):
                parts.append(str(payload["context"]))
            else:
                parts.append(json.dumps(payload, default=str)[:4000])
        elif payload is not None:
            parts.append(str(payload)[:4000])
        return "\n".join(p for p in parts if p) or "(no result)"


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
    # Human-readable argument summary, used by ``ToolRegistry.describe`` and in
    # reports. The authoritative machine-readable spec is ``args_schema``.
    parameters: dict[str, str] = {}
    #: Pydantic model describing this tool's arguments. Required for the native
    #: tool-calling path — it becomes the JSON schema the model fills in — and
    #: its field descriptions are what the model actually reads, so write them
    #: as instructions rather than labels.
    args_schema: "type[BaseModel] | None" = None

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

    def as_langchain_tool(self) -> BaseTool:
        """Adapt this tool for ``bind_tools`` / ``ToolNode``.

        The returned tool is coroutine-only: every ``AgentTool`` is async, and
        offering a sync entry point would mean either blocking an event loop or
        quietly spawning one. Callers on the native path are async already.
        """
        if self.args_schema is None:
            raise ValueError(
                f"Tool {self.name!r} declares no args_schema and cannot be bound "
                "for native tool calling."
            )

        async def _run(**kwargs) -> str:
            return (await self(**kwargs)).for_model()

        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            coroutine=_run,
            # Errors already come back inside ToolResult.for_model as text, so
            # LangChain's own exception handling has nothing left to catch.
            handle_tool_error=False,
        )


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

    def specs(self) -> list[dict]:
        return [tool.spec() for tool in self._tools.values()]

    def as_langchain_tools(self) -> list[BaseTool]:
        """Every bindable tool in this registry, for ``bind_tools``/``ToolNode``.

        Tools without an ``args_schema`` are skipped rather than raising: a tool
        can be perfectly usable on the direct path while not yet being worth
        exposing to the model, and one such tool must not make the whole
        registry unbindable.
        """
        bindable = []
        for tool in self._tools.values():
            if tool.args_schema is None:
                log.debug("Tool %s has no args_schema; not bindable", tool.name)
                continue
            bindable.append(tool.as_langchain_tool())
        return bindable

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
