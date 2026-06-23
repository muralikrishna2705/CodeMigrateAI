import logging
import re

from agents.base import AgentResult, BaseAgent
from models.state import MigrationState, MigrationType

log = logging.getLogger("CodeMigrateAI.MigratorAgent")


class MigratorAgent(BaseAgent):
    name = "MigratorAgent"
    requires_llm = True
    fast_mode_skip = False

    def __init__(self, llm_client, config: dict | None = None):
        super().__init__(llm_client, config)
        self.stream_callback = config.get("stream_callback") if config else None

    async def run(self, state: MigrationState) -> AgentResult:
        plan = state.migration_plan or {}
        metrics = state.code_metrics or {}
        is_upgrade = state.migration_type == MigrationType.UPGRADE_VERSION

        system_prompt = self._build_system_prompt(state, is_upgrade)
        context = self._build_context(plan, metrics)
        code_for_prompt = state.source_code[:4000]
        user_prompt = self._build_user_prompt(
            state, context, code_for_prompt, is_upgrade
        )

        if self.stream_callback:
            raw_output = ""
            async for token in self.llm.stream_llm(user_prompt, system_prompt):
                raw_output += token
                await self.stream_callback(token)
        else:
            raw_output = await self.llm.call_llm(user_prompt, system_prompt)

        cleaned_code = self._strip_fences(raw_output)

        if not cleaned_code.strip():
            raise ValueError("LLM returned empty code output")

        state.migrated_code = cleaned_code

        return AgentResult(
            success=True,
            summary=(
                f"Generated {len(cleaned_code.splitlines())} lines of "
                f"{state.target_language} {state.target_version}"
            ),
            details={
                "input_lines": metrics.get("total_lines"),
                "output_lines": len(cleaned_code.splitlines()),
                "migration_type": state.migration_type.value,
            },
        )

    def _build_system_prompt(self, state: MigrationState, is_upgrade: bool) -> str:
        if is_upgrade:
            return (
                "You are an expert "
                f"{state.source_language} developer who specialises in "
                "version migration. Your task is to upgrade "
                f"{state.source_language} code from version "
                f"{state.source_version} to {state.target_version}.\n"
            "Rules:\n"
            "  1. Output ONLY the migrated code — no explanations, "
            "no markdown fences\n"
            "  2. Preserve ALL original logic and business "
            "functionality exactly\n"
            "  3. Apply every transformation from the migration "
            "plan below\n"
            "  4. Add a short comment only where a migration "
            "change is non-obvious\n"
            "  5. Keep the same class/method structure unless "
            "restructuring is required"
        )
        return (
            "You are an expert polyglot software engineer. "
            "Your task is to convert "
            f"{state.source_language} code to idiomatic "
            f"{state.target_language} {state.target_version}.\n"
            "Rules:\n"
            "  1. Output ONLY the converted code — no explanations, "
            "no markdown fences\n"
            "  2. Preserve ALL original logic and business "
            "functionality exactly\n"
            "  3. Use idiomatic "
            f"{state.target_language} — do NOT do a "
            "word-for-word translation\n"
            "  4. Apply every transformation from the migration "
            "plan below\n"
            "  5. Use "
            f"{state.target_language} standard library equivalents "
            "throughout"
        )

    def _build_context(self, plan: dict, metrics: dict) -> str:
        context_lines = []
        if plan.get("strategy"):
            context_lines.append(f"Migration strategy: {plan['strategy']}")
        if plan.get("syntax_changes"):
            context_lines.append("Syntax transformations to apply:")
            context_lines.extend(f"  - {c}" for c in plan["syntax_changes"][:10])
        if plan.get("api_changes"):
            context_lines.append("API mappings to apply:")
            context_lines.extend(f"  - {a}" for a in plan["api_changes"][:10])
        if metrics.get("deprecated_patterns"):
            context_lines.append("Deprecated patterns to fix:")
            context_lines.extend(f"  - {d}" for d in metrics["deprecated_patterns"][:5])
        if plan.get("risk_areas"):
            context_lines.append("Handle these carefully:")
            context_lines.extend(f"  ⚠ {r}" for r in plan["risk_areas"][:5])
        return "\n".join(context_lines)

    def _build_user_prompt(
        self, state: MigrationState, context: str, code: str, is_upgrade: bool
    ) -> str:
        action = (
            f"Upgrade from {state.source_version} to {state.target_version}"
            if is_upgrade
            else f"Convert to {state.target_language} {state.target_version}"
        )
        return (
            f"{action}\n\n"
            f"{context}\n\n"
            f"SOURCE CODE ({state.source_language} {state.source_version}):\n"
            f"{code}\n\n"
            f"OUTPUT ({state.target_language} {state.target_version}) — "
            "only the code, nothing else:"
        )

    def _strip_fences(self, raw: str) -> str:
        fenced = re.match(r"^```[\w]*\n([\s\S]*?)```\s*$", raw.strip())
        if fenced:
            return fenced.group(1).strip()
        cleaned = re.sub(r"^```[\w]*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        cleaned = re.sub(
            r"(?i)^(here(?:'s| is) the (?:migrated|converted|upgraded) code[:\s]*\n)",
            "",
            cleaned,
        )
        return cleaned.strip()
