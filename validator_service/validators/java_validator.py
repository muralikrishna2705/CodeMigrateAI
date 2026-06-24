import re
from pathlib import Path

from .base import CommandSyntaxValidator


class JavaValidator(CommandSyntaxValidator):
    language = "java"
    tool = "javac"
    default_filename = "Main.java"

    def filename_for(self, code: str, version: str) -> str:
        match = re.search(r"\bpublic\s+(?:class|interface|enum|record)\s+(\w+)", code)
        if match:
            return f"{match.group(1)}.java"
        return self.default_filename

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return [
            "javac",
            "-Xlint:unchecked",
            "-Xlint:deprecation",
            str(source_path),
        ]
