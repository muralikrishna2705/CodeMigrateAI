from pathlib import Path

from .base import CommandSyntaxValidator


class CppValidator(CommandSyntaxValidator):
    language = "cpp"
    tool = "g++"
    default_filename = "main.cpp"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        standard = version if version in {"14", "17", "20", "23"} else "20"
        return [
            "g++",
            f"-std=c++{standard}",
            "-fsyntax-only",
            str(source_path),
        ]
