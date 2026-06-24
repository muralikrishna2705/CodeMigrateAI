from pathlib import Path

from .base import CommandSyntaxValidator


class GoValidator(CommandSyntaxValidator):
    language = "go"
    tool = "go"
    default_filename = "main.go"

    def prepare_code(self, code: str, version: str) -> str:
        if code.lstrip().startswith("package "):
            return code
        return f"package main\n\n{code}"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return ["go", "test", source_path.name]
