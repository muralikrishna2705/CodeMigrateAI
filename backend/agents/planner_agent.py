import logging

from config import get_settings
from llm.language_profiles import get_profile
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

        raw = await self.llm.call_llm(prompt, system_prompt)
        plan = self._extract_plan(raw)

        state.inline_plan = plan

        return AgentResult(
            success=True,
            summary=f"Generated {len(plan.splitlines())}-line migration plan",
            details={"plan": plan, "migration_type": migration_type.value},
        )

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
            "",
            "OUTPUT FORMAT (JSON only, no markdown):",
            "{",
            '  "plan_summary": "One-sentence overview",',
            '  "steps": [',
            '    {"step": 1, "action": "...", "details": "..."},',
            '    {"step": 2, "action": "...", "details": "..."}',
            "  ],",
            '  "risk_areas": ["area1", "area2"]',
            "}",
        ]
        return "\n".join(lines)

    def _extract_plan(self, raw: str) -> str:
        try:
            import json

            data = json.loads(raw)
            return data.get("plan_summary", raw.strip())
        except json.JSONDecodeError:
            if hasattr(self.llm, "extract_json"):
                try:
                    data = self.llm.extract_json(raw)
                    return data.get("plan_summary", raw.strip())
                except Exception:
                    pass
            return raw.strip()
