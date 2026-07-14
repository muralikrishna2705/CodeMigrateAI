import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ValidatorAgent")


class ValidatorAgent(BaseAgent):
    name = "ValidatorAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        validation_result = {"syntax_valid": True, "logic_check": None}

        if state.migrated_code:
            try:
                from validators import validate_syntax

                syntax_result = await validate_syntax(
                    state.migrated_code, state.target_language
                )
                validation_result["syntax_valid"] = syntax_result.valid
                validation_result["syntax_errors"] = syntax_result.to_dict()
            except Exception as e:
                log.warning("Syntax validation failed: %s", e)
                validation_result["syntax_valid"] = False
                validation_result["syntax_errors"] = str(e)

        state.validation_result = validation_result

        return AgentResult(
            success=True,
            summary=(
                f"Syntax validation: "
                f"{'passed' if validation_result['syntax_valid'] else 'failed'}"
            ),
            details=validation_result,
        )
