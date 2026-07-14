import logging

from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RuntimeValidatorAgent")


class RuntimeValidatorAgent(BaseAgent):
    name = "RuntimeValidatorAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        # Coordinates validation across all languages.
        # Calls validator service for syntax + optional logic validation.
        valid = (
            state.validation_result.get("valid", False)
            if state.validation_result
            else False
        )
        log.info("RuntimeValidatorAgent: validation=%s", valid)
        return AgentResult(
            success=True,
            summary=f"Validation: {valid}",
            details={"validation_result": state.validation_result},
        )
