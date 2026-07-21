import logging

from config import get_settings
from llm.language_profiles import get_profile
from models.schemas import MigrationPlan
from models.state import MigrationState, MigrationType

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.PlannerAgent")


class PlannerAgent(BaseAgent):
    name = "PlannerAgent"
    requires_llm = True

    async def run(self, state: MigrationState) -> AgentResult:
        source_profile = get_profile(state.source_language)
        target_profile = get_profile(state.target_language)

        migration_type = self._detect_migration_type(state)

        prompt = self._build_planning_prompt(
            source_profile, target_profile, state, migration_type
        )
        system_prompt = self._build_system_prompt(migration_type, state)

        # Planning is structured reasoning, not code generation — routed to the
        # fast model. The step structure exists to make the model think in steps;
        # downstream consumes the rendered prose, not the shape.
        plan = await self._plan(prompt, system_prompt)

        # Self-reflection (Dimension 3): critique the plan and refine it once if the
        # model is not confident in it. Opt-in and best-effort — off by default and
        # any failure keeps the original plan.
        reflection_note = None
        if get_settings().enable_reflection:
            plan, reflection_note = await self._reflect_and_refine(
                plan, state, prompt, system_prompt
            )

        state.inline_plan = plan

        details = {"plan": plan, "migration_type": migration_type.value}
        if reflection_note:
            details["reflection"] = reflection_note
        return AgentResult(
            success=True,
            summary=f"Generated {len(plan.splitlines())}-line migration plan",
            details=details,
        )

    async def _reflect_and_refine(
        self, plan: str, state: MigrationState, planning_prompt: str, system_prompt: str
    ) -> tuple[str, dict]:
        """Reflect on the plan; regenerate once when confidence is low.

        Returns ``(plan, note)`` — the refined plan if a low-confidence critique
        produced actionable feedback, otherwise the original. The note surfaces the
        verdict in the agent's report.
        """
        from agents.tools.reflection import PLAN_CRITERIA

        reflection = await self.reflect(
            state, plan, criteria=PLAN_CRITERIA, stage="migration plan"
        )
        note = {
            "confidence": reflection.confidence,
            "recommendation": reflection.recommendation,
        }

        settings = get_settings()
        should_refine = (
            not reflection.passed
            and reflection.confidence < settings.reflection_min_confidence
            and bool(reflection.feedback)
        )
        if not should_refine:
            return plan, note

        refine_prompt = (
            f"{planning_prompt}\n\n---\n\nYour previous plan was:\n{plan}\n\n"
            f"A reviewer flagged these issues:\n{reflection.feedback}\n\n"
            "Produce an improved plan that addresses every point."
        )
        refined = await self._plan(refine_prompt, system_prompt)

        if refined and refined.strip():
            note["refined"] = True
            return refined, note
        return plan, note

    def _detect_migration_type(self, state: MigrationState) -> MigrationType:
        from llm.language_profiles import ProfileRegistry

        source = ProfileRegistry.normalize(state.source_language)
        target = ProfileRegistry.normalize(state.target_language)
        if source == target:
            return MigrationType.UPGRADE_VERSION
        return MigrationType.CONVERT_LANGUAGE

    def _build_system_prompt(
        self, migration_type: MigrationType, state: MigrationState
    ) -> str:
        if migration_type == MigrationType.UPGRADE_VERSION:
            return (
                f"You are a senior {state.target_language} modernization architect. "
                "Produce a step-by-step plan for upgrading code. Be specific."
            )
        return (
            f"You are a senior migration architect converting {state.source_language} "
            f"to {state.target_language}. Produce a detailed step-by-step migration plan."
        )

    def _build_planning_prompt(
        self, source_profile, target_profile, state, migration_type
    ) -> str:
        settings = get_settings()
        lines = [
            "MIGRATION PLANNING TASK",
            f"Source: {source_profile.display_name} {state.source_version}",
            f"Target: {target_profile.display_name} {state.target_version}",
            f"Type: {'Version upgrade' if migration_type == MigrationType.UPGRADE_VERSION else 'Language conversion'}",
            "",
            "Analyze the source code below and produce a migration plan.",
            "For each step, describe what changes are needed and why.",
            "",
            "SOURCE CODE:",
            "```" + source_profile.language_id,
            state.source_code[: settings.max_llm_code_chars],
            "```",
        ]
        return "\n".join(lines)

    async def _plan(self, prompt: str, system_prompt: str) -> str:
        """Produce a rendered plan, or an empty string when none can be had.

        Empty is meaningful: ``run`` keeps the previous plan on a failed refine,
        and an empty first plan simply leaves ``inline_plan`` unset — the
        MigratorAgent already handles a missing plan section.
        """
        result = await self._call_structured(MigrationPlan, prompt, system_prompt)
        return result.render() if result else ""
