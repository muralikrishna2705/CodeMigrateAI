async def validate_syntax(code: str, language: str):
    from models.validation import SyntaxError as SyntaxDiagnostic
    from models.validation import ValidationResult

    if language == "python":
        try:
            import ast

            ast.parse(code)
            return ValidationResult(valid=True, errors=[], warnings=[])
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[
                    SyntaxDiagnostic(
                        line=e.lineno or 0, column=e.offset or 0, message=e.msg
                    )
                ],
                warnings=[],
            )

    # All other languages: delegate to the external validator service, which has
    # real compilers (javac, node --check, tsc, dotnet build, go test, kotlinc,
    # rustc, g++) for the remaining 8 languages.
    from clients.validator_client import ValidatorClient
    from config import get_settings

    settings = get_settings()
    validator = ValidatorClient(
        base_url=settings.validator_url,
        timeout_sec=settings.validator_timeout_sec,
    )
    try:
        result = await validator.validate(code=code, language=language, version="")
        return ValidationResult(
            valid=result.get("valid", False),
            errors=[
                SyntaxDiagnostic(
                    line=e.get("line", 0),
                    column=e.get("column", 0),
                    message=e.get("message", ""),
                )
                for e in result.get("errors", [])
            ],
            warnings=[
                SyntaxDiagnostic(
                    line=w.get("line", 0),
                    column=w.get("column", 0),
                    message=w.get("message", ""),
                )
                for w in result.get("warnings", [])
            ],
        )
    except Exception as exc:
        return ValidationResult(
            valid=False,
            errors=[],
            warnings=[
                SyntaxDiagnostic(
                    line=0,
                    column=0,
                    message=f"Validator service unavailable: {exc}",
                )
            ],
        )
    finally:
        await validator.close()
