from pathlib import Path

from .base import CommandSyntaxValidator


class TypeScriptValidator(CommandSyntaxValidator):
    language = "typescript"
    tool = "tsc"
    default_filename = "main.ts"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return [
            "tsc",
            "--noEmit",
            "--strict",
            "--skipLibCheck",
            str(source_path),
        ]
