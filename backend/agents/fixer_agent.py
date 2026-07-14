import logging
import re

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.FixerAgent")


class FixerAgent(BaseAgent):
    name = "FixerAgent"
    requires_llm = True

    async def run(self, state: MigrationState) -> AgentResult:
        settings = get_settings()
        validation = state.validation_result or {}
        errors = validation.get("errors", [])
        warnings = validation.get("warnings", [])

        if not errors:
            if validation.get("valid") is False:
                log.warning(
                    "Validation failed with zero errors; likely a service issue"
                )
                return AgentResult(
                    success=True,
                    summary=(
                        "Validation service returned no actionable errors; "
                        "skipping fix"
                    ),
                    details={"validation_result": validation},
                )
            return AgentResult(success=True, summary="No errors to fix")

        error_text = "\n".join(
            f"Line {e.get('line', '?')}:{e.get('column', '?')} {e.get('message', '?')}"
            for e in (errors + warnings)
        )

        prompt = (
            f"The following {state.target_language} code failed validation.\n\n"
            f"ERRORS:\n{error_text}\n\n"
            f"CODE:\n```{state.target_language}\n"
            f"{state.migrated_code[:settings.max_llm_code_chars]}\n```\n\n"
            "Fix each error. Output ONLY the corrected code, no explanations, "
            "no markdown fences."
        )

        fixed = await self.llm.call_llm(
            prompt,
            system_prompt=(
                f"You are a {state.target_language} compiler engineer. Fix the "
                "validation errors. Output only the corrected source code."
            ),
        )

        # Strip markdown fences if the model added them anyway.
        fenced = re.match(r"^```[\w+-]*\n([\s\S]*?)```\s*$", fixed.strip())
        if fenced:
            fixed = fenced.group(1).strip()

        state.migrated_code = fixed
        return AgentResult(
            success=True,
            summary=f"Fixed {len(errors)} validation error(s)",
            details={"errors_fixed": len(errors), "code_length": len(fixed)},
        )
