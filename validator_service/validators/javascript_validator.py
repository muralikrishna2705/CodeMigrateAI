from pathlib import Path

from .base import CommandSyntaxValidator


class JavaScriptValidator(CommandSyntaxValidator):
    language = "javascript"
    tool = "node"
    default_filename = "main.js"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return ["node", "--check", str(source_path)]
