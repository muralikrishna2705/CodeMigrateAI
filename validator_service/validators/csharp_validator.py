import re
import textwrap
from pathlib import Path

from .base import CommandSyntaxValidator, ValidationResult

# A type declaration makes the snippet a complete compilation unit on its own.
_TYPE_DECL = re.compile(r"\b(class|struct|record|interface|enum)\s+\w+")
# A member carrying an access modifier can only live inside a type — it is not
# a legal top-level statement, so a snippet that has one but no enclosing type
# must be wrapped.
_ACCESS_MODIFIED_MEMBER = re.compile(r"^\s*(public|private|protected|internal)\b", re.MULTILINE)


class CSharpValidator(CommandSyntaxValidator):
    language = "csharp"
    tool = "dotnet"
    default_filename = "Program.cs"

    async def validate(self, code: str, version: str) -> ValidationResult:
        stripped = code.strip()
        has_type = bool(_TYPE_DECL.search(stripped))
        # net8.0 with ImplicitUsings compiles both full type declarations and
        # bare top-level-statement programs as-is; wrap ONLY the case that is
        # otherwise uncompilable: access-modified members with no enclosing type.
        if not has_type and _ACCESS_MODIFIED_MEMBER.search(stripped):
            code = f"public class Program\n{{\n{code}\n}}"
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
