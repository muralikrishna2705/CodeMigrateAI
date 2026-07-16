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

        issues = errors + warnings
        code = state.migrated_code
        error_text = "\n".join(
            f"Line {e.get('line', '?')}:{e.get('column', '?')} {e.get('message', '?')}"
            for e in issues
        )

        prompt = self._build_prompt(state, code, error_text, issues, settings)
        fixed = await self.llm.call_llm(
            prompt,
            system_prompt=(
                f"You are a {state.target_language} compiler engineer. Fix the "
                "validation errors at the indicated lines. Use only the standard "
                "library and APIs shown in the reference examples — do not invent "
                "new dependencies. Output only the corrected source code."
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

    def _build_prompt(self, state, code, error_text, issues, settings) -> str:
        """Assemble a targeted fix prompt.

        Beyond the raw error list, this localizes each error to its offending
        source line(s) and reuses the retrieved reference examples + analyzer
        summary, so the model corrects in place with grounded APIs instead of
        guessing from errors alone.
        """
        parts = [
            f"The following {state.target_language} code failed validation.",
            "",
            f"ERRORS:\n{error_text}",
        ]

        annotated = self._annotate_error_sites(code.splitlines(), issues)
        if annotated:
            parts += ["", f"OFFENDING LINES:\n{annotated}"]

        summary = (state.code_metrics or {}).get("summary")
        if summary:
            parts += ["", f"SOURCE CONTEXT: {summary}"]

        # Reuse the retrieved reference examples so the fix stays grounded in real
        # target-version APIs rather than inventing new ones.
        if state.rag_context:
            parts += ["", state.rag_context.strip()[: settings.max_llm_code_chars]]

        parts += [
            "",
            f"CODE:\n```{state.target_language}\n"
            f"{code[: settings.max_llm_code_chars]}\n```",
            "",
            "Fix each error. Output ONLY the corrected code, no explanations, "
            "no markdown fences.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _annotate_error_sites(code_lines, issues, context: int = 2) -> str:
        """Render the offending lines (± ``context``) with 1-based line numbers."""
        if not code_lines:
            return ""
        wanted: set[int] = set()
        for issue in issues:
            line = issue.get("line")
            if not isinstance(line, int) or line < 1:
                continue
            for ln in range(line - context, line + context + 1):
                if 1 <= ln <= len(code_lines):
                    wanted.add(ln)
        if not wanted:
            return ""
        return "\n".join(f"{ln:>4}: {code_lines[ln - 1]}" for ln in sorted(wanted))
