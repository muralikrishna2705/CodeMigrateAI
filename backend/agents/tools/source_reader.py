"""SourceReaderTool — read regions of the *full* source code on demand.

This is the reframed CodeReaderTool. The original spec described reading files
from disk, but a migration's input is a single ``source_code`` string carried on
the state — there is no project tree to walk.

The real problem it solves instead: ``settings.max_llm_code_chars`` (4000) caps
what reaches the model, so on any sizeable input every agent reasons about a
truncated prefix and simply cannot see the tail. This tool reads the *untruncated*
string — an outline of its symbols, a specific line range, or the neighbourhood
of a pattern — so an agent can pull in the part it actually needs rather than
whatever happened to fall inside the first 4000 characters.

Stateless: callers pass ``source_code`` explicitly, since one shared tool
instance serves every concurrent migration.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from agents.tools.base import AgentTool, ToolResult


class SourceReaderArgs(BaseModel):
    source_code: str = Field(description="The full source text to read from.")
    mode: Literal["outline", "read", "find"] = Field(
        default="outline",
        description=(
            "'outline' lists every class/function; 'read' returns a line range; "
            "'find' locates a symbol and its surrounding lines."
        ),
    )
    start_line: int = Field(default=1, description="First line, 1-indexed. mode=read.")
    end_line: int = Field(default=0, description="Last line, inclusive. mode=read.")
    pattern: str = Field(default="", description="Symbol or text to locate. mode=find.")
    context_lines: int = Field(
        default=3, description="Lines of context around each match. mode=find."
    )

# Definition sites across the nine supported languages. Deliberately loose — an
# outline that over-reports a few lines is far cheaper than one that misses the
# function an agent was looking for.
_SYMBOL_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:public|private|protected|internal|static|final|abstract|async|export|pub)\s+)*"
    r"(?:"
    r"class\s+(?P<cls>\w+)"
    r"|(?:def|func|fn|fun|sub|function)\s+(?P<func>\w+)"
    r"|(?:interface|struct|enum|trait|impl|record|type)\s+(?P<typ>\w+)"
    r")",
    re.MULTILINE,
)

_MAX_SNIPPET_LINES = 400


class SourceReaderTool(AgentTool):
    name = "source_reader"
    description = (
        "Read the full source code beyond the truncation limit: get an outline "
        "of its classes/functions, read a specific line range, or find the lines "
        "around a symbol. Use when the code is long and you need to see a part "
        "that was cut off."
    )
    args_schema = SourceReaderArgs
    parameters = {
        "source_code": "the full source text to read from",
        "mode": "one of: outline, read, find",
        "start_line": "first line (1-indexed) — mode=read",
        "end_line": "last line (inclusive) — mode=read",
        "pattern": "symbol or text to locate — mode=find",
        "context_lines": "lines of context around each match — mode=find",
    }

    async def run(
        self,
        source_code: str = "",
        mode: str = "outline",
        start_line: int = 1,
        end_line: int = 0,
        pattern: str = "",
        context_lines: int = 10,
        **_,
    ) -> ToolResult:
        if not source_code:
            return ToolResult(
                tool=self.name, success=False, error="source_code is required"
            )

        lines = source_code.splitlines()
        if mode == "outline":
            return self._outline(source_code, lines)
        if mode == "read":
            return self._read(lines, start_line, end_line)
        if mode == "find":
            return self._find(lines, pattern, context_lines)
        return ToolResult(
            tool=self.name,
            success=False,
            error=f"unknown mode {mode!r}; expected outline, read, or find",
        )

    def _outline(self, source_code: str, lines: list[str]) -> ToolResult:
        symbols = []
        for match in _SYMBOL_PATTERN.finditer(source_code):
            name = match.group("cls") or match.group("func") or match.group("typ")
            if not name:
                continue
            line_no = source_code.count("\n", 0, match.start()) + 1
            symbols.append({"name": name, "line": line_no, "text": lines[line_no - 1].strip()})

        rendered = "\n".join(f"L{s['line']}: {s['text']}" for s in symbols)
        return ToolResult(
            tool=self.name,
            success=True,
            data={
                "total_lines": len(lines),
                "total_chars": len(source_code),
                "symbols": symbols,
                "outline": rendered,
            },
            summary=f"{len(symbols)} symbol(s) across {len(lines)} lines",
        )

    def _read(self, lines: list[str], start_line: int, end_line: int) -> ToolResult:
        start = max(1, int(start_line or 1))
        end = int(end_line) if end_line else len(lines)
        end = min(max(end, start), len(lines))
        # Bound the read so "read the whole thing" can't re-introduce the very
        # context blowout the truncation limit exists to prevent.
        if end - start + 1 > _MAX_SNIPPET_LINES:
            end = start + _MAX_SNIPPET_LINES - 1

        if start > len(lines):
            return ToolResult(
                tool=self.name,
                success=False,
                error=f"start_line {start} is past end of source ({len(lines)} lines)",
            )

        snippet = "\n".join(lines[start - 1 : end])
        return ToolResult(
            tool=self.name,
            success=True,
            data={"start_line": start, "end_line": end, "snippet": snippet},
            summary=f"Read lines {start}-{end} of {len(lines)}",
        )

    def _find(self, lines: list[str], pattern: str, context_lines: int) -> ToolResult:
        if not pattern:
            return ToolResult(
                tool=self.name, success=False, error="pattern is required for mode=find"
            )

        context = max(0, min(int(context_lines or 0), 50))
        needle = pattern.lower()
        matches = []
        for index, line in enumerate(lines):
            if needle not in line.lower():
                continue
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            matches.append(
                {
                    "line": index + 1,
                    "snippet": "\n".join(lines[start:end]),
                }
            )
            if len(matches) >= 5:
                break

        return ToolResult(
            tool=self.name,
            success=True,
            data={"pattern": pattern, "matches": matches},
            summary=(
                f"{len(matches)} match(es) for {pattern!r}"
                if matches
                else f"No match for {pattern!r}"
            ),
        )
