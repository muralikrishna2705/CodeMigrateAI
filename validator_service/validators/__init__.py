from .base import SyntaxValidator
from .cpp_validator import CppValidator
from .csharp_validator import CSharpValidator
from .go_validator import GoValidator
from .java_validator import JavaValidator
from .javascript_validator import JavaScriptValidator
from .kotlin_validator import KotlinValidator
from .python_validator import PythonValidator
from .rust_validator import RustValidator
from .typescript_validator import TypeScriptValidator

_VALIDATORS: dict[str, SyntaxValidator] = {
    "java": JavaValidator(),
    "python": PythonValidator(),
    "javascript": JavaScriptValidator(),
    "typescript": TypeScriptValidator(),
    "csharp": CSharpValidator(),
    "go": GoValidator(),
    "kotlin": KotlinValidator(),
    "rust": RustValidator(),
    "cpp": CppValidator(),
}

_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "c#": "csharp",
    "cs": "csharp",
    "c++": "cpp",
    "golang": "go",
}


def normalize(language: str) -> str:
    key = language.strip().lower()
    return _ALIASES.get(key, key)


def get_validator(language: str) -> SyntaxValidator | None:
    return _VALIDATORS.get(normalize(language))


def supported_languages() -> list[str]:
    return sorted(_VALIDATORS)


__all__ = ["get_validator", "supported_languages"]
