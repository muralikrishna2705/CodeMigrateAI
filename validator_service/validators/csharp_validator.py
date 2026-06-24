import textwrap
from pathlib import Path

from .base import CommandSyntaxValidator, ValidationResult


class CSharpValidator(CommandSyntaxValidator):
    language = "csharp"
    tool = "dotnet"
    default_filename = "Program.cs"

    async def validate(self, code: str, version: str) -> ValidationResult:
        top_level_markers = (
            "using ",
            "namespace ",
            "public ",
            "internal ",
            "class ",
        )
        if not code.lstrip().startswith(top_level_markers):
            code = f"public class Program {{\n{code}\n}}"
        return await super().validate(code, version)

    def command_args(self, source_path: Path, version: str) -> list[str]:
        project = source_path.parent / "Validation.csproj"
        project.write_text(
            textwrap.dedent(
                """
                <Project Sdk="Microsoft.NET.Sdk">
                  <PropertyGroup>
                    <TargetFramework>net8.0</TargetFramework>
                    <ImplicitUsings>enable</ImplicitUsings>
                    <Nullable>enable</Nullable>
                  </PropertyGroup>
                </Project>
                """
            ).strip(),
            encoding="utf-8",
        )
        return ["dotnet", "build", "--nologo", "-v:q"]
