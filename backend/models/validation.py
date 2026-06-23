from dataclasses import dataclass


@dataclass
class SyntaxError:
    line: int
    column: int
    message: str
    severity: str = "error"


@dataclass
class ValidationResult:
    valid: bool
    errors: list[SyntaxError]
    warnings: list[SyntaxError]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [
                {
                    "line": e.line,
                    "column": e.column,
                    "message": e.message,
                    "severity": e.severity,
                }
                for e in self.errors
            ],
            "warnings": [
                {
                    "line": w.line,
                    "column": w.column,
                    "message": w.message,
                    "severity": w.severity,
                }
                for w in self.warnings
            ],
        }


@dataclass
class LogicCheckResult:
    equivalent: bool
    differences: list[str]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "equivalent": self.equivalent,
            "differences": self.differences,
            "confidence": self.confidence,
        }
