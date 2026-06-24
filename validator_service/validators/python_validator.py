import ast

from .base import SyntaxValidator, ValidationIssue, ValidationResult


class PythonValidator(SyntaxValidator):
    language = "python"

    async def validate(self, code: str, version: str) -> ValidationResult:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationIssue(
                        line=exc.lineno or 0,
                        column=exc.offset or 0,
                        message=exc.msg,
                    )
                ],
            )
        return ValidationResult(valid=True)
