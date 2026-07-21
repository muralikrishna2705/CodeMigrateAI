"""Pydantic schemas for every structured LLM response.

These replace ~250 lines of regex JSON salvage that existed because the old
stack asked a 1.3b model for JSON in prose and then repaired whatever came back.
A schema handed to ``BaseChatModel.with_structured_output`` is enforced by the
provider's constrained decoder instead, so the malformed cases stop existing
rather than being cleaned up after the fact.

Design constraints worth knowing before adding a schema here:

- **Stick to str / int / float / bool / Literal / list / nested models.**
  Provider schema dialects are narrower than JSON Schema; free-form
  ``dict[str, Any]`` fields in particular are unsupported or silently dropped by
  some backends. A field that must hold arbitrary keys belongs in a nested model
  with named fields, or as a serialized string.
- **Every field needs a default.** A model that omits an optional key should
  yield a usable object, not a ValidationError — these are enrichment payloads,
  and a partial analysis beats no analysis.
- **Descriptions are prompt surface.** They are serialized into the schema the
  model sees, so they do real work; write them as instructions.
"""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

# The three actions a critique can recommend. Shared so the reflection tool, the
# ReflectorAgent, and graph.conditions.reflect_condition cannot drift apart.
Recommendation = Literal["pass", "re-generate", "gather-more-info"]


class MigrationOutput(BaseModel):
    """The MigratorAgent's result: the converted code plus a one-line summary."""

    plan_summary: str = Field(
        default="",
        description="One concise sentence describing the changes you made.",
    )
    # The aliases are not decoration. On the streaming path the model is bound to
    # a response schema but the JSON is parsed by us as it arrives, and models
    # reach for `code`/`output` often enough that accepting them costs nothing
    # and saves a whole regeneration round trip.
    migrated_code: str = Field(
        default="",
        validation_alias=AliasChoices(
            "migrated_code", "code", "output", "result", "source_code"
        ),
        description=(
            "The COMPLETE migrated version of the source code, as a single "
            "string. Migrate the actual source given — never placeholder or "
            "example code."
        ),
    )

    model_config = {"populate_by_name": True}


class SemanticAnalysis(BaseModel):
    """AnalyzerAgent's LLM pass, layered over the static metrics.

    Every field defaults to empty because ``PromptComposer`` and
    ``RAGPipeline._metric_terms`` read these keys unconditionally — a missing key
    would be an AttributeError three layers away from the model that omitted it.
    """

    deprecated_patterns: list[str] = Field(
        default_factory=list,
        description="Constructs deprecated or removed in the target version.",
    )
    migration_challenges: list[str] = Field(
        default_factory=list,
        description="Specific things that make this migration hard.",
    )
    key_constructs: list[str] = Field(
        default_factory=list,
        description="Language constructs central to this code (generics, async, …).",
    )
    summary: str = Field(default="", description="One-paragraph analysis summary.")


class DeepAnalysis(BaseModel):
    """DeepAnalyzerAgent's structural pass for high-complexity code."""

    complex_constructs: list[str] = Field(
        default_factory=list,
        description="Generics, reflection, async/await, macros, metaprogramming.",
    )
    stdlib_dependencies: list[str] = Field(
        default_factory=list,
        description="Standard-library modules that need a target equivalent.",
    )
    inheritance_depth: int = Field(
        default=0, description="Deepest inheritance chain in the code."
    )
    design_patterns: list[str] = Field(default_factory=list)
    breaking_changes: list[str] = Field(
        default_factory=list,
        description="Changes that will alter behaviour if translated naively.",
    )
    recommended_strategy: str = Field(
        default="", description="How to approach this migration, in one paragraph."
    )


class PlanStep(BaseModel):
    step: int = Field(default=0)
    action: str = Field(default="", description="What to change.")
    details: str = Field(default="", description="How, and why it is needed.")


class MigrationPlan(BaseModel):
    """PlannerAgent's step-by-step plan."""

    plan_summary: str = Field(default="", description="One-sentence overview.")
    steps: list[PlanStep] = Field(default_factory=list)
    risk_areas: list[str] = Field(
        default_factory=list, description="Parts most likely to break."
    )

    def render(self) -> str:
        """Flatten to the plain-text plan the downstream prompt expects.

        ``state.inline_plan`` is injected into the migration prompt as prose, so
        the structure exists to make the model *think* in steps, not to be
        consumed structurally downstream.
        """
        lines = [self.plan_summary] if self.plan_summary else []
        lines += [
            f"{s.step}. {s.action}" + (f" — {s.details}" if s.details else "")
            for s in self.steps
        ]
        if self.risk_areas:
            lines.append("Risk areas: " + ", ".join(self.risk_areas))
        return "\n".join(line for line in lines if line.strip())


class Critique(BaseModel):
    """A self-critique of one piece of output (the Reflexion signal).

    ``confidence`` is bounded by the schema rather than clamped after the fact,
    which is what lets ``_coerce_confidence``'s manual clamping go away.
    """

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="0.0 = badly wrong, 1.0 = correct and complete.",
    )
    recommendation: Recommendation = Field(
        default="pass",
        description=(
            "'pass' = good enough to proceed; 're-generate' = has real problems, "
            "redo it; 'gather-more-info' = cannot be sure without more reference "
            "material."
        ),
    )
    feedback: str = Field(
        default="",
        description="Specific, actionable issues to fix. Empty when passing.",
    )


class RouteDecision(BaseModel):
    """DispatcherAgent's routing choice."""

    deep_analyze: bool = Field(
        description="True if this code needs a deep structural analysis pass."
    )
    reasoning: str = Field(default="", description="One sentence justifying the call.")


class SubTaskPlan(BaseModel):
    """OrchestratorAgent's decomposition into concurrently-runnable sub-tasks.

    Constrained to the known task names by ``Literal`` so a hallucinated task
    cannot reach ``graph.subgraphs.resolve_tasks`` — previously this was a
    post-hoc filter against PARALLELIZABLE_TASKS.
    """

    tasks: list[Literal["analysis", "retrieval"]] = Field(
        default_factory=list,
        description=(
            "Preparation sub-tasks to run before planning. 'analysis' = deep "
            "structural analysis; 'retrieval' = fetch reference documentation."
        ),
    )


class RelevanceGrade(BaseModel):
    """A retrieved document graded against the query (CRAG / Self-RAG)."""

    relevant: bool = Field(description="Does this document help answer the query?")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How relevant, 0.0 to 1.0."
    )
    reason: str = Field(default="")


class SubQueries(BaseModel):
    """Reformulated or decomposed retrieval queries (multi-query, multi-hop)."""

    queries: list[str] = Field(
        default_factory=list,
        description="Standalone search queries, each useful on its own.",
    )


class RetrievalRouteDecision(BaseModel):
    """Which retrieval strategy to use for this specific query.

    This is the switch that makes retrieval agentic rather than configured: the
    model picks per query instead of every run reading one static setting.
    """

    strategy: Literal[
        "single_hop",
        "hyde",
        "multi_query",
        "multi_hop",
        "corrective",
        "parent_document",
    ] = Field(
        default="single_hop",
        description=(
            "'single_hop' = the query is specific and well-formed; 'hyde' = the "
            "query is vague and would match better against a hypothetical answer; "
            "'multi_query' = the query is ambiguous and worth paraphrasing; "
            "'multi_hop' = answering needs several dependent lookups; "
            "'corrective' = grounding accuracy matters more than speed; "
            "'parent_document' = surrounding context matters more than the snippet."
        ),
    )
    needs_retrieval: bool = Field(
        default=True,
        description=(
            "False when the model can answer from the code alone — skipping "
            "retrieval entirely is the cheapest possible path."
        ),
    )
    reasoning: str = Field(default="")
