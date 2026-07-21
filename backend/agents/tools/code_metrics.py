"""CodeMetricsTool — static code metrics, computed on demand.

The metric logic itself is lifted verbatim from ``AnalyzerAgent._compute_static_metrics``
so the numbers are unchanged. What changes is *who* can ask for them: previously
only the analyzer's fixed pre-pass computed metrics, once, over the whole source.
Now any agent can measure any snippet at any point — e.g. the analyzer measuring
a single hot function it pulled with the source reader.

Deliberately still called deterministically by ``AnalyzerAgent`` on every run:
``code_metrics`` is load-bearing for DispatcherAgent routing (``complexity``),
RAG query terms (``RAGPipeline._metric_terms``), and PromptComposer. Making it
on-demand-only would let those silently lose their input whenever the model
declined to call the tool.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from agents.tools.base import AgentTool, ToolResult


class CodeMetricsArgs(BaseModel):
    code: str = Field(description="The source code to measure.")

_BRANCH_PATTERN = re.compile(
    r"\b(if|else|elif|for|while|switch|case|catch|except|try)\b"
)
_CLASS_PATTERN = re.compile(r"\bclass\s+\w+")
_METHOD_PATTERN = re.compile(r"\b(def|void|func|fn|fun|sub|function)\s+\w+\s*\(")
_IMPORT_PATTERN = re.compile(r"\b(import|require|include|using|from)\b")

# Branch-count thresholds separating low/medium/high complexity. Unchanged from
# the analyzer's original inline rule — DispatcherAgent routes on the result.
_MEDIUM_COMPLEXITY_BRANCHES = 6
_HIGH_COMPLEXITY_BRANCHES = 25


class CodeMetricsTool(AgentTool):
    name = "code_metrics"
    description = (
        "Compute static metrics (lines, classes, methods, imports, branch count, "
        "complexity) for a code snippet. Use to size up code before deciding how "
        "much analysis it needs."
    )
    args_schema = CodeMetricsArgs
    parameters = {"code": "the source code to measure"}

    async def run(self, code: str = "", **_) -> ToolResult:
        metrics = self.compute(code)
        return ToolResult(
            tool=self.name,
            success=True,
            data=metrics,
            summary=(
                f"{metrics['total_lines']} lines · "
                f"{metrics['class_count']} classes · "
                f"{metrics['method_count']} methods · "
                f"complexity={metrics['complexity']}"
            ),
        )

    @staticmethod
    def compute(code: str) -> dict[str, Any]:
        """Pure metric computation, callable without the async tool wrapper."""
        lines = code.splitlines()
        branches = len(_BRANCH_PATTERN.findall(code))
        classes = len(_CLASS_PATTERN.findall(code))
        methods = len(_METHOD_PATTERN.findall(code))
        imports = len(_IMPORT_PATTERN.findall(code))

        if branches < _MEDIUM_COMPLEXITY_BRANCHES:
            complexity = "low"
        elif branches < _HIGH_COMPLEXITY_BRANCHES:
            complexity = "medium"
        else:
            complexity = "high"

        return {
            "total_lines": len(lines),
            "non_empty_lines": len([ln for ln in lines if ln.strip()]),
            "branch_count": branches,
            "class_count": classes,
            "method_count": methods,
            "import_count": imports,
            "complexity": complexity,
        }
