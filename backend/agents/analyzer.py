import logging
import re
from typing import Any

from llm.prompts import ANALYZER_PROMPT
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.AnalyzerAgent")


class AnalyzerAgent(BaseAgent):
    name = "AnalyzerAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        code = state.source_code

        static_metrics = self._compute_static_metrics(code)

        if self.config.get("enable_semantic_analysis"):
            try:
                semantic = await self._llm_semantic_analysis(state, code)
                metrics = {**static_metrics, **semantic}
            except Exception as e:
                log.warning("LLM semantic analysis failed: %s", e)
                metrics = {
                    **static_metrics,
                    "deprecated_patterns": [],
                    "migration_challenges": [],
                    "key_constructs": [],
                    "summary": (
                        f"Static analysis: {len(code.splitlines())} lines "
                        f"of {state.source_language} code."
                    ),
                }
        else:
            metrics = {
                **static_metrics,
                "deprecated_patterns": [],
                "migration_challenges": [],
                "key_constructs": [],
                "summary": (
                    f"Static analysis: {len(code.splitlines())} lines "
                    f"of {state.source_language} code."
                ),
            }

        state.code_metrics = metrics

        return AgentResult(
            success=True,
            summary=(
                f"{metrics['total_lines']} lines · "
                f"{metrics['class_count']} classes · "
                f"{metrics['method_count']} methods · "
                f"complexity={metrics['complexity']}"
            ),
            details=metrics,
        )

    def _compute_static_metrics(self, code: str) -> dict[str, Any]:
        lines = code.splitlines()
        branches = len(
            re.findall(
                r"\b(if|else|elif|for|while|switch|case|catch|except|try)\b", code
            )
        )
        classes = len(re.findall(r"\bclass\s+\w+", code))
        methods = len(
            re.findall(r"\b(def|void|func|fn|fun|sub|function)\s+\w+\s*\(", code)
        )
        imports = len(re.findall(r"\b(import|require|include|using|from)\b", code))

        complexity = (
            "low" if branches < 6 else "medium" if branches < 25 else "high"
        )

        return {
            "total_lines": len(lines),
            "non_empty_lines": len([ln for ln in lines if ln.strip()]),
            "branch_count": branches,
            "class_count": classes,
            "method_count": methods,
            "import_count": imports,
            "complexity": complexity,
        }

    async def _llm_semantic_analysis(
        self, state: MigrationState, code: str
    ) -> dict:
        prompt = ANALYZER_PROMPT.format(
            source_language=state.source_language,
            source_version=state.source_version,
            target_language=state.target_language,
            target_version=state.target_version,
            code=code[:3500],
        )

        raw = await self.llm.call_llm(
            prompt,
            system_prompt=(
                "You are a code analysis expert. "
                "Respond ONLY with valid JSON."
            ),
        )
        return self.llm.extract_json(raw)
