"""Agent tool layer — the callables agents invoke on demand.

``build_registry`` is the single place tools are constructed and gated by
settings, so wiring a new tool is one entry here rather than an edit in every
consumer. ``graph.nodes`` calls it once at startup and registers the result on
the Provider, from which agents receive it via ``needs = ("tools",)``.
"""

from agents.tools.base import AgentTool, ToolCall, ToolRegistry, ToolResult
from agents.tools.code_metrics import CodeMetricsTool
from agents.tools.reflection import ReflectionTool
from agents.tools.semantic_search import SemanticSearchTool
from agents.tools.source_reader import SourceReaderTool
from agents.tools.syntax_checker import SyntaxCheckTool
from agents.tools.vector_db import VectorDBTool
from agents.tools.web_search import WebSearchTool

__all__ = [
    "AgentTool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "CodeMetricsTool",
    "ReflectionTool",
    "SemanticSearchTool",
    "SourceReaderTool",
    "SyntaxCheckTool",
    "VectorDBTool",
    "WebSearchTool",
    "build_registry",
]


def build_registry(
    rag_pipeline=None, migration_memory=None, llm_client=None, settings=None
) -> ToolRegistry:
    """Construct the enabled tools.

    Tools whose backing dependency is missing are skipped rather than registered
    in a broken state: an agent asking for an unregistered tool gets a clean
    "not available" ToolResult, whereas a registered-but-dead tool would fail on
    every call. ``tools_enabled=False`` yields an empty registry, which every
    agent already handles — that is the kill switch.
    """
    from config import get_settings

    settings = settings or get_settings()
    if not settings.tools_enabled:
        return ToolRegistry()

    tools: list[AgentTool] = []
    timeout = settings.tool_timeout_sec

    if settings.tool_code_metrics_enabled:
        tools.append(CodeMetricsTool(timeout_sec=timeout))
    if settings.tool_syntax_check_enabled:
        tools.append(SyntaxCheckTool())  # keeps its own longer compile timeout
    if settings.tool_source_reader_enabled:
        tools.append(SourceReaderTool(timeout_sec=timeout))
    if settings.tool_vector_db_enabled and rag_pipeline is not None:
        tools.append(VectorDBTool(rag_pipeline, timeout_sec=timeout))
    if settings.tool_web_search_enabled:
        tools.append(WebSearchTool())  # keeps its own network timeout
    if settings.tool_semantic_search_enabled and migration_memory is not None:
        tools.append(SemanticSearchTool(migration_memory, timeout_sec=timeout))
    # The reflect_output tool needs an LLM to critique with, and it only earns a
    # slot in the catalog when reflection is switched on — an extra tool a small
    # model might mis-select is a cost, so it stays out of the default catalog.
    if settings.enable_reflection and llm_client is not None:
        tools.append(ReflectionTool(llm_client, timeout_sec=timeout))

    return ToolRegistry(tools)
