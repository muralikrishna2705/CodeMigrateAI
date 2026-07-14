import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.ValidatorAgent")


class ValidatorAgent(BaseAgent):
    name = "ValidatorAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        validation_result = {"valid": True, "errors": [], "warnings": []}

        if state.migrated_code:
            try:
                from validators import validate_syntax

                syntax_result = await validate_syntax(
                    state.migrated_code, state.target_language, state.target_version
                )
                validation_result["valid"] = syntax_result.valid
                result_dict = syntax_result.to_dict()
                validation_result["errors"] = result_dict.get("errors", [])
                validation_result["warnings"] = result_dict.get("warnings", [])
            except Exception as e:
                log.warning("Syntax validation failed: %s", e)
                validation_result["valid"] = False
                validation_result["errors"] = [
                    {"line": 0, "column": 0, "message": str(e)}
                ]

        state.validation_result = validation_result

        return AgentResult(
            success=True,
            summary=(
                f"Syntax validation: "
                f"{'passed' if validation_result['valid'] else 'failed'}"
            ),
            details=validation_result,
        )
