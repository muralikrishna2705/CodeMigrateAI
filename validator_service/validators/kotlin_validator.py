from pathlib import Path

from .base import CommandSyntaxValidator


class KotlinValidator(CommandSyntaxValidator):
    language = "kotlin"
    tool = "kotlinc"
    default_filename = "main.kt"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return ["kotlinc", str(source_path), "-d", "out.jar"]
