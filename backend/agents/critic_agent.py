"""CriticAgent — specialized self-critique for migrated code.

Where the generic ``BaseAgent.reflect`` hook judges any output, the CriticAgent
knows it is looking at *migrated code* and grades it on the three things a
migration most often gets wrong: behavior preservation, idiomatic target-language
usage, and completeness. It pairs the LLM critique with a cheap heuristic scan for
the tell-tale signs of an incomplete migration (stubbed bodies, ``TODO`` markers,
``NotImplementedError``), because a 1.3b model will often rate its own stub as
"complete" — the heuristics are the floor that catches what the model misses.

It is used two ways: the :class:`~agents.reflector_agent.ReflectorAgent` (the
graph ``reflect`` node) delegates code reflection to it, and the MigratorAgent
runs it inline to decide whether to regenerate before ever leaving the node.
"""

import logging
import re

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent, ReflectionResult

log = logging.getLogger("CodeMigrateAI.CriticAgent")

# Markers that betray an incomplete/stubbed migration across the supported
# languages. Matched case-insensitively against whole lines. Deliberately narrow
# (high signal / low false-positive): each is something a *finished* migration
# almost never contains.
_STUB_MARKERS = (
    r"\bTODO\b",
    r"\bFIXME\b",
    r"NotImplementedError",
    r"UnsupportedOperationException",
    r"todo!\(",  # Rust
    r"unimplemented!\(",  # Rust
    r"panic\(\s*[\"']not implemented",  # Go
    r"raise\s+NotImplementedError",  # Python
    r"throw\s+new\s+NotImplementedException",  # C#
    r"\.\.\.\s*(#.*)?$",  # a bare ellipsis body
    r"placeholder",
)
_STUB_RE = re.compile("|".join(_STUB_MARKERS), re.IGNORECASE)


class CriticAgent(BaseAgent):
    name = "CriticAgent"
    requires_llm = True
    needs = ("tools",)

    async def run(self, state: MigrationState) -> AgentResult:
        """Critique ``state.migrated_code`` and record the verdict on the state."""
        result = await self.critique(
            state.migrated_code,
            state=state,
            extra_context=self._failure_context(state),
        )
        state.reflection_score = result.confidence
        state.reflection_feedback = result.feedback
        state.reflection_recommendation = result.recommendation
        return AgentResult(
            success=True,
            summary=(
                f"Critique: {result.recommendation} "
                f"(confidence {result.confidence:.2f})"
            ),
            details={
                "confidence": result.confidence,
                "recommendation": result.recommendation,
                "feedback": result.feedback,
                **(result.details or {}),
            },
        )

    async def critique(
        self, code: str, *, state: MigrationState, extra_context: str = ""
    ) -> ReflectionResult:
        """Grade ``code`` on correctness/idiom/completeness, floored by heuristics.

        Returns a passing :class:`ReflectionResult` when there is nothing to judge.
        The heuristic stub scan can turn an LLM "pass" into a "re-generate" (the
        model under-reports its own stubs) but never the reverse — a clean scan
        leaves the model's verdict untouched.
        """
        from agents.tools.reflection import CODE_CRITERIA, evaluate_output

        if not (code or "").strip():
            return ReflectionResult.neutral()

        stubs = self._detect_stubs(code)

        context = self._compose_context(state, extra_context, stubs)
        # Prefer the reflect_output tool when present (so critique flows through
        # the tool interface); fall back to the shared engine directly otherwise.
        tool = self.tools.get("reflect_output")
        if tool is not None:
            tr = await self._call_tool(
                "reflect_output",
                output=code,
                criteria=list(CODE_CRITERIA),
                stage=f"{state.target_language} {state.target_version} migrated code",
                context=context,
            )
            if tr.success and isinstance(tr.data, dict):
                result = ReflectionResult(
                    confidence=tr.data.get("confidence", 1.0),
                    recommendation=tr.data.get("recommendation", "pass"),
                    feedback=tr.data.get("feedback", ""),
                    details=tr.data.get("details"),
                )
            else:
                result = await evaluate_output(
                    self.llm,
                    output=code,
                    criteria=CODE_CRITERIA,
                    stage=f"{state.target_language} {state.target_version} migrated code",
                    context=context,
                )
        else:
            result = await evaluate_output(
                self.llm,
                output=code,
                criteria=CODE_CRITERIA,
                stage=f"{state.target_language} {state.target_version} migrated code",
                context=context,
            )

        return self._apply_stub_floor(result, stubs)

    @staticmethod
    def _detect_stubs(code: str) -> list[str]:
        """Return the distinct stub/incompleteness markers found in ``code``."""
        found: list[str] = []
        for line in code.splitlines():
            match = _STUB_RE.search(line)
            if match:
                snippet = match.group(0).strip()
                if snippet and snippet not in found:
                    found.append(snippet)
        return found

    def _apply_stub_floor(
        self, result: ReflectionResult, stubs: list[str]
    ) -> ReflectionResult:
        """Force a re-generate verdict when the code is visibly incomplete.

        A migration that still contains ``TODO``/``NotImplementedError``/a stubbed
        body is incomplete by definition, regardless of how confident the model
        is — so this overrides an over-optimistic "pass".
        """
        if not stubs:
            return result

        note = "Incomplete migration — remove stubs/placeholders: " + ", ".join(
            stubs[:5]
        )
        feedback = f"{result.feedback}\n{note}".strip() if result.feedback else note
        # Cap confidence and escalate a "pass" to "re-generate"; keep a stronger
        # existing recommendation (e.g. the model already said gather-more-info).
        recommendation = (
            "re-generate" if result.recommendation == "pass" else result.recommendation
        )
        details = dict(result.details or {})
        details["stub_markers"] = stubs
        return ReflectionResult(
            confidence=min(result.confidence, 0.4),
            recommendation=recommendation,
            feedback=feedback,
            details=details,
        )

    def _compose_context(
        self, state: MigrationState, extra_context: str, stubs: list[str]
    ) -> str:
        parts = []
        summary = (state.code_metrics or {}).get("summary")
        if summary:
            parts.append(f"Source intent: {summary}")
        if extra_context:
            parts.append(extra_context)
        if stubs:
            parts.append(
                "A pre-scan already flagged these incompleteness markers: "
                + ", ".join(stubs[:5])
            )
        return "\n".join(parts)

    @staticmethod
    def _failure_context(state: MigrationState) -> str:
        """Fold prior validation errors into the critique (Reflexion signal).

        Reflecting *with* the record of what already failed is the point of the
        pattern — it turns a blind re-rating into targeted feedback.
        """
        validation = state.validation_result or {}
        errors = validation.get("errors") or []
        if not errors:
            return ""
        lines = [
            f"Line {e.get('line', '?')}: {e.get('message', '?')}" for e in errors[:5]
        ]
        return "Prior validation errors:\n" + "\n".join(lines)

    def should_run(self, state: MigrationState) -> bool:
        # Only meaningful once there is code to critique.
        return bool(state.migrated_code and get_settings().enable_reflection)
