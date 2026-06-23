async def validate_syntax(code: str, language: str):
    from models.validation import SyntaxError, ValidationResult

    if language == "python":
        try:
            import ast
            ast.parse(code)
            return ValidationResult(valid=True, errors=[], warnings=[])
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[
                    SyntaxError(
                        line=e.lineno or 0, column=e.offset or 0, message=e.msg
                    )
                ],
                warnings=[],
            )

    if language == "java":
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=[
                SyntaxError(
                    line=0,
                    column=0,
                    message="Java validation not implemented - requires javac",
                )
            ],
        )

    return ValidationResult(
        valid=True,
        errors=[],
        warnings=[
            SyntaxError(
                line=0, column=0, message=f"No validator for {language}"
            )
        ],
    )
