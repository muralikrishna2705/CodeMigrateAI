"""ReflectionTool — the on-demand self-critique primitive, plus its shared engine.

``evaluate_output`` is the single place the reflection LLM prompt lives: it asks a
model to rate a piece of generated output against a set of criteria and recommend
an action. ``BaseAgent.reflect`` calls it, the :class:`CriticAgent` and
:class:`ReflectorAgent` call it, and :class:`ReflectionTool` wraps it so an agent
can reflect *mid-generation* through the same tool interface it uses for
everything else (``self._call_tool("reflect_output", ...)``).

Why prompt-driven and not native tool calling: like the rest of this stack, the
model is Ollama's ``/api/generate``, which has no ``tools`` parameter — so the
critique is a constrained-JSON completion, and every failure mode (no LLM, model
down, unparseable answer) degrades to a *passing* verdict. Reflection improves a
migration when it works and is invisible when it can't, but never blocks one.
"""

import json
import logging

from agents.base import ReflectionResult
from agents.tools.base import AgentTool, ToolResult

log = logging.getLogger("CodeMigrateAI.Reflection")

# Criteria presets, shared so the in-agent reflection and the graph nodes judge by
# the same yardstick. Callers may pass their own list.
GENERIC_CRITERIA = ("correctness", "completeness", "clarity")
CODE_CRITERIA = (
    "behavior preservation (the migrated code must do exactly what the source did)",
    "idiomatic use of the target language and version",
    "completeness (no omitted logic, stubbed bodies, or TODO placeholders)",
)
PLAN_CRITERIA = (
    "completeness (covers every construct that must change)",
    "feasibility of each step",
    "risk coverage (calls out what could break)",
)

_VALID_RECOMMENDATIONS = {"pass", "re-generate", "gather-more-info"}


def _fast_model_kwargs(llm) -> dict:
    """Route the critique to the fast model when the client exposes one.

    Reflection is evaluation, not code generation, so it does not need the
    heavyweight coder model. Mirrors ``BaseAgent._fast_model_kwargs`` but works
    off a bare client (the tool and the engine both hold a client, not an agent).
    """
    fast = getattr(llm, "fast_model", None)
    return {"model": fast} if fast else {}


def _coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, conf))


def _normalize_recommendation(value) -> str:
    """Map a free-text recommendation onto the three canonical actions.

    A small model rarely echoes the exact enum, so we accept near-misses
    ("regenerate", "needs more context") and fall back to ``"pass"`` for anything
    unrecognized — the safe default that keeps the migration moving.
    """
    text = str(value or "").strip().lower()
    if text in _VALID_RECOMMENDATIONS:
        return text
    if "regen" in text or "redo" in text or "rewrite" in text:
        return "re-generate"
    if "info" in text or "context" in text or "retriev" in text or "gather" in text:
        return "gather-more-info"
    return "pass"


def _parse(raw: str, llm) -> dict | None:
    """Parse the critique JSON, tolerating fenced or prefixed output."""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        extract_json = getattr(llm, "extract_json", None)
        if extract_json is None:
            return None
        try:
            data = extract_json(raw)
        except Exception:  # noqa: BLE001 — parsing is best-effort
            return None
    return data if isinstance(data, dict) else None


async def evaluate_output(
    llm,
    *,
    output: str,
    criteria=GENERIC_CRITERIA,
    stage: str = "output",
    context: str = "",
) -> ReflectionResult:
    """Ask ``llm`` to self-critique ``output`` and recommend an action.

    Returns a passing :meth:`ReflectionResult.neutral` when there is nothing to
    judge (empty output) or no usable LLM, so callers never need to guard the
    degraded path themselves.
    """
    call_llm = getattr(llm, "call_llm", None)
    if call_llm is None or not (output or "").strip():
        return ReflectionResult.neutral()

    criteria_str = (
        "; ".join(criteria) if isinstance(criteria, (list, tuple)) else str(criteria)
    )
    prompt_parts = [
        f"Critically review the following {stage}. Judge it against these "
        f"criteria: {criteria_str}.",
    ]
    if context:
        prompt_parts += ["", f"CONTEXT:\n{context}"]
    prompt_parts += [
        "",
        f"{stage.upper()}:",
        output[:6000],
        "",
        "Rate your confidence from 0.0 (badly wrong) to 1.0 (correct and "
        "complete), and choose ONE recommendation:",
        "- \"pass\": good enough to proceed.",
        "- \"re-generate\": has real problems; redo it.",
        "- \"gather-more-info\": can't be sure without more reference material.",
        "",
        'Respond with ONLY JSON: {"confidence": <0.0-1.0>, "recommendation": '
        '"pass|re-generate|gather-more-info", "feedback": "specific, actionable '
        'issues to fix (empty if pass)"}',
    ]
    prompt = "\n".join(prompt_parts)

    try:
        raw = await call_llm(
            prompt,
            system_prompt=(
                "You are a meticulous senior reviewer performing self-reflection. "
                "Be honest about flaws. Respond ONLY with the JSON object."
            ),
            fmt="json",
            **_fast_model_kwargs(llm),
        )
    except Exception as exc:  # noqa: BLE001 — reflection is strictly optional
        log.warning("Reflection call failed: %s", exc)
        return ReflectionResult.neutral()

    data = _parse(raw, llm)
    if not data:
        log.info("Reflection response unparseable; treating as pass")
        return ReflectionResult.neutral()

    confidence = _coerce_confidence(data.get("confidence", 1.0))
    recommendation = _normalize_recommendation(data.get("recommendation", "pass"))
    feedback = str(data.get("feedback", "") or "").strip()
    return ReflectionResult(
        confidence=confidence,
        recommendation=recommendation,
        feedback=feedback,
        details={"stage": stage, "criteria": list(criteria)},
    )


class ReflectionTool(AgentTool):
    """Reflect on a piece of generated output on demand.

    Constructed with the shared LLM client (``build_registry`` wires it when
    reflection is enabled), so an agent can critique a candidate before committing
    to it — the mid-generation counterpart to the post-hoc ``reflect`` graph node.
    """

    name = "reflect_output"
    description = (
        "Self-critique a piece of generated output against quality criteria. "
        "Returns a 0.0-1.0 confidence, a recommendation (pass / re-generate / "
        "gather-more-info), and actionable feedback."
    )
    parameters = {
        "output": "the text/code to critique",
        "criteria": "what to judge it against (optional)",
        "stage": "what kind of output it is, e.g. 'migrated code' (optional)",
        "context": "extra context such as prior errors (optional)",
    }

    def __init__(self, llm_client, timeout_sec: float | None = None) -> None:
        super().__init__(timeout_sec=timeout_sec)
        self.llm = llm_client

    async def run(
        self,
        output: str = "",
        criteria=GENERIC_CRITERIA,
        stage: str = "output",
        context: str = "",
        **_,
    ) -> ToolResult:
        result = await evaluate_output(
            self.llm,
            output=output,
            criteria=criteria or GENERIC_CRITERIA,
            stage=stage,
            context=context,
        )
        return ToolResult(
            tool=self.name,
            success=True,
            data={
                "confidence": result.confidence,
                "recommendation": result.recommendation,
                "feedback": result.feedback,
                "details": result.details,
            },
            summary=(
                f"Reflection: {result.recommendation} "
                f"(confidence {result.confidence:.2f})"
            ),
        )
