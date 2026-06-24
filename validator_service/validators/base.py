import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    line: int = 0
    column: int = 0
    message: str
    severity: str = "error"


class ValidationResult(BaseModel):
    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class SyntaxValidator:
    language = ""

    async def validate(self, code: str, version: str) -> ValidationResult:
        raise NotImplementedError


class CommandSyntaxValidator(SyntaxValidator):
    tool = ""
    default_filename = "Main.txt"

    async def validate(self, code: str, version: str) -> ValidationResult:
        if shutil.which(self.tool) is None:
            return self._tool_missing_result()

        with tempfile.TemporaryDirectory(prefix=f"codemigrate-{self.language}-") as td:
            workdir = Path(td)
            filename = self.filename_for(code, version)
            source_path = workdir / filename
            source_path.write_text(self.prepare_code(code, version), encoding="utf-8")
            return await self.run_command(
                self.command_args(source_path, version),
                workdir,
            )

    def filename_for(self, code: str, version: str) -> str:
        return self.default_filename

    def prepare_code(self, code: str, version: str) -> str:
        return code

    def command_args(self, source_path: Path, version: str) -> list[str]:
        raise NotImplementedError

    async def run_command(
        self,
        args: list[str],
        cwd: Path,
    ) -> ValidationResult:
        timeout = float(os.getenv("VALIDATOR_TIMEOUT", "30"))
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationIssue(
                        message=(
                            f"{self.language} validation timed out after "
                            f"{timeout:g}s"
                        )
                    )
                ],
            )

        output = "\n".join(
            part.decode("utf-8", errors="replace").strip()
            for part in (stderr, stdout)
            if part
        ).strip()
        if proc.returncode == 0:
            warnings = [self._issue(output, "warning")] if output else []
            return ValidationResult(valid=True, errors=[], warnings=warnings)

        return ValidationResult(
            valid=False,
            errors=[
                self._issue(output or f"{self.tool} exited with {proc.returncode}")
            ],
            warnings=[],
        )

    def _tool_missing_result(self) -> ValidationResult:
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=[
                ValidationIssue(
                    message=(
                        f"{self.tool} is not installed in the validator service; "
                        f"{self.language} syntax validation was skipped."
                    ),
                    severity="warning",
                )
            ],
        )

    def _issue(self, output: str, severity: str = "error") -> ValidationIssue:
        line = 0
        column = 0
        location = re.search(r":(\d+)(?::(\d+))?", output)
        if location:
            line = int(location.group(1))
            if location.group(2):
                column = int(location.group(2))

        first_line = (
            output.strip().splitlines()[0] if output.strip() else "Validation failed"
        )
        return ValidationIssue(
            line=line,
            column=column,
            message=first_line[:1000],
            severity=severity,
        )
