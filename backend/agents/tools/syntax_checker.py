"""SyntaxCheckTool — syntax validation on demand.

Wraps the same ``validators.validate_syntax`` entry point the ValidatorAgent has
always used (Python via ``ast``, the other eight languages via the external
validator service), so results are identical. The gain is that validation is no
longer only a post-hoc gate at the end of the graph: the MigratorAgent can check
a candidate *before* committing to it, and the FixerAgent can confirm a fix
without a full graph loop.

``ValidatorAgent`` still calls this unconditionally on every run — its
``validation_result`` drives ``graph.conditions.validate_condition`` and the
whole fix loop, so it must never depend on a model choosing to call it.
"""

from pydantic import BaseModel, Field

from agents.tools.base import AgentTool, ToolResult


class SyntaxCheckArgs(BaseModel):
    code: str = Field(description="The code to validate.")
    language: str = Field(description="Target language, e.g. python, java, go.")
    version: str = Field(default="", description="Target language version.")


class SyntaxCheckTool(AgentTool):
    name = "syntax_check"
    description = (
        "Validate that code is syntactically correct for a target language and "
        "version. Returns validity plus any error diagnostics with line numbers."
    )
    args_schema = SyntaxCheckArgs
    parameters = {
        "code": "the code to validate",
        "language": "target language (e.g. python, java, go)",
        "version": "target language version (optional)",
    }

    # The non-Python path calls the external validator service over the network;
    # give it room to compile without inheriting the tighter default.
    timeout_sec = 45.0

    async def run(
        self, code: str = "", language: str = "", version: str = "", **_
    ) -> ToolResult:
        if not code.strip():
            return ToolResult(
                tool=self.name,
                success=True,
                data={"valid": True, "errors": [], "warnings": []},
                summary="No code to validate",
            )

        from validators import validate_syntax

        # A validator failure is a *result* (invalid code), not a tool failure —
        # mirroring ValidatorAgent's original except branch, which recorded the
        # exception as a syntax error rather than failing the agent.
        try:
            syntax_result = await validate_syntax(code, language, version)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool=self.name,
                success=True,
                data={
                    "valid": False,
                    "errors": [{"line": 0, "column": 0, "message": str(exc)}],
                    "warnings": [],
                },
                summary=f"Syntax validation errored: {exc}",
            )

        result_dict = syntax_result.to_dict()
        data = {
            "valid": syntax_result.valid,
            "errors": result_dict.get("errors", []),
            "warnings": result_dict.get("warnings", []),
        }
        return ToolResult(
            tool=self.name,
            success=True,
            data=data,
            summary=(
                f"Syntax validation: {'passed' if data['valid'] else 'failed'}"
                f"{f' ({len(data['errors'])} error(s))' if data['errors'] else ''}"
            ),
        )
