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

import logging

from llm.structured import coerce_or_none
from models.schemas import Critique
from pydantic import BaseModel, Field

from agents.base import ReflectionResult
from agents.tools.base import AgentTool, ToolResult


class ReflectArgs(BaseModel):
    output: str = Field(description="The text or code to critique.")
    criteria: str = Field(
        default="", description="What to judge it against, comma-separated."
    )
    stage: str = Field(
        default="output", description="What kind of output it is, e.g. 'migrated code'."
    )
    context: str = Field(
        default="", description="Extra context such as prior errors."
    )

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


def _fast_model_kwargs(llm) -> dict:
    """Route the critique to the fast model when the client exposes one.

    Reflection is evaluation, not code generation, so it does not need the
    heavyweight coder model. Mirrors ``BaseAgent._fast_model_kwargs`` but works
    off a bare client (the tool and the engine both hold a client, not an agent).
    """
    fast = getattr(llm, "fast_model", None)
    return {"model": fast} if fast else {}


async def _critique(llm, prompt: str, system_prompt: str) -> Critique | None:
    """Get a :class:`Critique` from ``llm``, natively when it can.

    Mirrors ``BaseAgent._call_structured`` but works off a bare client, because
    the engine and the tool both hold a client rather than an agent. Returns None
    on any failure — the caller turns that into a *passing* verdict.
    """
    chat_model = getattr(llm, "chat_model", None)
    if chat_model is not None:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            model = chat_model("fast").with_structured_output(Critique)
            return await model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            )
        except Exception as exc:  # noqa: BLE001 — reflection is strictly optional
            log.warning("Structured reflection failed: %s", exc)
            return None

    call_llm = getattr(llm, "call_llm", None)
    if call_llm is None:
        return None
    try:
        raw = await call_llm(
            prompt, system_prompt=system_prompt, fmt="json", **_fast_model_kwargs(llm)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Reflection call failed: %s", exc)
        return None
    return coerce_or_none(Critique, raw)


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
    usable = getattr(llm, "chat_model", None) or getattr(llm, "call_llm", None)
    if usable is None or not (output or "").strip():
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
        # What each recommendation means lives in the Critique schema's field
        # descriptions, which are serialized into the schema the model sees —
        # repeating them here would be two copies free to drift apart.
        "Rate your confidence honestly and choose one recommendation.",
    ]
    prompt = "\n".join(prompt_parts)

    critique = await _critique(
        llm,
        prompt,
        "You are a meticulous senior reviewer performing self-reflection. "
        "Be honest about flaws.",
    )
    if critique is None:
        log.info("No usable critique; treating as pass")
        return ReflectionResult.neutral()

    # The schema bounds confidence to [0.0, 1.0] and constrains recommendation to
    # the three canonical actions, so the manual clamping and the fuzzy
    # "regenerate"/"needs more context" string matching that used to live here
    # are no longer reachable states.
    return ReflectionResult(
        confidence=critique.confidence,
        recommendation=critique.recommendation,
        feedback=critique.feedback.strip(),
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
    args_schema = ReflectArgs
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
