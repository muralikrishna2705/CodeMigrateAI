import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ValidatorAgent")


class ValidatorAgent(BaseAgent):
    name = "ValidatorAgent"
    requires_llm = False
    needs = ("tools",)

    async def run(self, state: MigrationState) -> AgentResult:
        validation_result = {"valid": True, "errors": [], "warnings": []}

        if state.migrated_code:
            validation_result = await self._validate(state)

        state.validation_result = validation_result

        return AgentResult(
            success=True,
            summary=(
                f"Syntax validation: "
                f"{'passed' if validation_result['valid'] else 'failed'}"
            ),
            # The report carries the tool call for observability; `state.validation_result`
            # stays the bare {valid, errors, warnings} shape that validate_condition
            # and the FixerAgent read.
            details={**validation_result, "tool_calls": self.tool_call_log()},
        )

    async def _validate(self, state: MigrationState) -> dict:
        """Validate via the tool, falling back to the validator directly.

        This agent calls the tool unconditionally rather than letting the model
        decide: ``validation_result`` drives ``graph.conditions.validate_condition``
        and therefore the entire fix loop. If validation only happened when a
        model chose to request it, a skipped call would look exactly like a pass
        and ship unvalidated code.

        The fallback covers ``tools_enabled=False``, where the tool is absent but
        the fix loop still needs a verdict.
        """
        result = await self._call_tool(
            "syntax_check",
            code=state.migrated_code,
            language=state.target_language,
            version=state.target_version,
        )
        if result.success and result.data:
            return result.data

        log.debug("syntax_check tool unavailable (%s); validating inline", result.error)
        try:
            from validators import validate_syntax

            syntax_result = await validate_syntax(
                state.migrated_code, state.target_language, state.target_version
            )
            result_dict = syntax_result.to_dict()
            return {
                "valid": syntax_result.valid,
                "errors": result_dict.get("errors", []),
                "warnings": result_dict.get("warnings", []),
            }
        except Exception as e:
            log.warning("Syntax validation failed: %s", e)
            return {
                "valid": False,
                "errors": [{"line": 0, "column": 0, "message": str(e)}],
                "warnings": [],
            }
