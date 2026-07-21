import logging
from typing import Any

from config import get_settings
from llm.prompts import ANALYZER_PROMPT
from models.schemas import SemanticAnalysis
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent
from agents.tools.code_metrics import CodeMetricsTool

log = logging.getLogger("CodeMigrateAI.AnalyzerAgent")


class AnalyzerAgent(BaseAgent):
    name = "AnalyzerAgent"
    requires_llm = False
    needs = ("tools",)

    async def run(self, state: MigrationState) -> AgentResult:
        code = state.source_code

        static_metrics = await self._static_metrics(code)

        # Defaults sit *under* the semantic output rather than replacing it: an
        # LLM that returns partial JSON (a small model routinely omits a key)
        # would otherwise leave code_metrics missing fields that PromptComposer
        # and RAGPipeline._metric_terms read unconditionally.
        metrics = {**static_metrics, **self._empty_semantics(state, code)}
        if self._semantic_enabled():
            try:
                metrics.update(await self._llm_semantic_analysis(state, code))
            except Exception as e:
                log.warning("LLM semantic analysis failed: %s", e)

        state.code_metrics = metrics

        return AgentResult(
            success=True,
            summary=(
                f"{metrics['total_lines']} lines · "
                f"{metrics['class_count']} classes · "
                f"{metrics['method_count']} methods · "
                f"complexity={metrics['complexity']}"
            ),
            details={**metrics, "tool_calls": self.tool_call_log()},
        )

    def _semantic_enabled(self) -> bool:
        """Whether to run the optional LLM semantic pass.

        Explicit config wins (the pipeline registry passes it, and tests set it
        directly); otherwise fall back to the setting. The settings fallback is
        what makes this work on the graph path at all: the Runtime builds each
        agent's config from its declared ``needs``, which never included this
        flag, so ``config.get("enable_semantic_analysis")`` was always None there
        and the semantic pass had been silently dead since the DI refactor —
        regardless of ``enable_semantic_analysis`` being True by default.
        """
        if "enable_semantic_analysis" in self.config:
            return bool(self.config["enable_semantic_analysis"])
        return bool(get_settings().enable_semantic_analysis)

    async def _static_metrics(self, code: str) -> dict[str, Any]:
        """Metrics via the tool, falling back to computing them directly.

        The fallback is not defensive padding: ``code_metrics`` feeds
        DispatcherAgent routing, the RAG query terms, and PromptComposer, so it
        must exist on every run. When tools are disabled (``tools_enabled=False``)
        the tool is simply absent, and the analyzer still has to produce metrics.
        """
        result = await self._call_tool("code_metrics", code=code)
        if result.success and result.data:
            return result.data
        log.debug("code_metrics tool unavailable (%s); computing inline", result.error)
        return CodeMetricsTool.compute(code)

    @staticmethod
    def _empty_semantics(state: MigrationState, code: str) -> dict[str, Any]:
        """The static-only shape. Keys must always be present — RAGPipeline's
        ``_metric_terms`` and the PromptComposer read them unconditionally."""
        return {
            "deprecated_patterns": [],
            "migration_challenges": [],
            "key_constructs": [],
            "summary": (
                f"Static analysis: {len(code.splitlines())} lines "
                f"of {state.source_language} code."
            ),
        }

    async def _llm_semantic_analysis(self, state: MigrationState, code: str) -> dict:
        settings = get_settings()
        prompt = ANALYZER_PROMPT.format(
            source_language=state.source_language,
            source_version=state.source_version,
            target_language=state.target_language,
            target_version=state.target_version,
            code=code[: settings.max_llm_code_chars],
        )

        # On-demand: when the code is longer than what fits in the prompt, ask the
        # source reader for a map of the whole file. Without this the model sees
        # only the first max_llm_code_chars and reports on a truncated prefix as
        # if it were the whole program — silently missing every construct past
        # the cut. The outline is small enough to append and names what was lost.
        outline = await self._outline_if_truncated(code, settings.max_llm_code_chars)
        if outline:
            prompt = f"{prompt}\n\nFULL-FILE OUTLINE (code above is truncated):\n{outline}"

        result = await self._call_structured(
            SemanticAnalysis,
            prompt,
            system_prompt="You are a code analysis expert.",
        )
        # None means the semantic pass produced nothing usable. Returning {} lets
        # the caller's dict merge leave the static-only defaults in place, which
        # is exactly the degraded shape those defaults exist for.
        return result.model_dump() if result else {}

    async def _outline_if_truncated(self, code: str, limit: int) -> str:
        if len(code) <= limit:
            return ""
        result = await self._call_tool(
            "source_reader", source_code=code, mode="outline"
        )
        if not result.success or not result.data:
            return ""
        return result.data.get("outline", "")
