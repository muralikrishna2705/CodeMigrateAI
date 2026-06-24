from pathlib import Path

from .base import CommandSyntaxValidator


class RustValidator(CommandSyntaxValidator):
    language = "rust"
    tool = "rustc"
    default_filename = "main.rs"

    def command_args(self, source_path: Path, version: str) -> list[str]:
        return ["rustc", "--crate-type=lib", "--emit=metadata", str(source_path)]
