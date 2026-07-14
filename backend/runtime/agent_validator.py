import logging

from config import get_settings
from models.state import MigrationState

from agents.base import AgentResult, BaseAgent

log = logging.getLogger("CodeMigrateAI.RuntimeValidatorAgent")


class RuntimeValidatorAgent(BaseAgent):
    name = "RuntimeValidatorAgent"
    requires_llm = False

    async def run(self, state: MigrationState) -> AgentResult:
        settings = get_settings()
        if not settings.enable_validation or not state.migrated_code:
            return AgentResult(
                success=True,
                summary="Validation skipped (disabled or no code)",
            )

        from clients.validator_client import ValidatorClient

        validator = ValidatorClient(
            base_url=settings.validator_url,
            timeout_sec=settings.validator_timeout_sec,
        )
        try:
            result = await validator.validate(
                code=state.migrated_code,
                language=state.target_language,
                version=state.target_version,
            )
            state.validation_result = result
            valid = result.get("valid", False)
            log.info(
                "RuntimeValidatorAgent: %s → valid=%s", state.target_language, valid
            )
            return AgentResult(
                success=True,
                summary=f"Validation: {'PASS' if valid else 'FAIL'}",
                details={
                    "valid": valid,
                    "error_count": len(result.get("errors", [])),
                    "warning_count": len(result.get("warnings", [])),
                },
            )
        except Exception as exc:
            log.warning("Validator service unavailable: %s", exc)
            return AgentResult(
                success=True,
                summary=f"Validator unavailable: {exc}",
                details={"error": str(exc)},
            )
        finally:
            await validator.close()
