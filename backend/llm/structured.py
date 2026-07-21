"""Getting a validated Pydantic object out of a model, whatever the model is.

There are two ways a schema gets filled here:

**Native** — a real chat model supports ``with_structured_output``, so the
provider constrains decoding to the schema and returns a validated instance.
This is the production path and it cannot produce malformed output.

**Salvage** — the object standing in for an LLM is a test double (or any client
without ``chat_model``) that returns a JSON string. :func:`coerce` parses it,
tolerating the ways a model garnishes JSON, and validates against the schema.

The salvage code below is the last surviving copy of what used to be duplicated
across six agents, the reflection tool, and the RAG strategies. It lives in one
place now, and production doesn't execute it.
"""

import json
import logging
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("CodeMigrateAI.Structured")

T = TypeVar("T", bound=BaseModel)

# Conversational lead-ins models prepend to JSON despite instructions not to.
_PREAMBLE = re.compile(
    r"(?i)^(?:here(?:'s| is) (?:the |my |your )?"
    r"(?:migrated|converted|upgraded) code[:\s]*|output[:\s]*|"
    r"result[:\s]*|sure[^.]*\.)"
)
_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def salvage_json(raw: str) -> dict:
    """Best-effort extraction of one JSON object from noisy model output.

    Tries, in order: the whole string, a markdown-fenced block, the longest
    balanced ``{…}`` span, and a brace-repair for output truncated mid-object.
    Raises ``ValueError`` when none of it yields an object.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response")

    cleaned = _PREAMBLE.sub("", text).strip()

    for candidate in (cleaned, text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    for block in _FENCE.findall(cleaned):
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Balanced-brace scan: collect every top-level {...} span, prefer the
    # longest (the outer object rather than a nested fragment).
    spans: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(cleaned[start : i + 1])
                start = -1

    for span in sorted(spans, key=len, reverse=True):
        try:
            data = json.loads(span)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Truncated output: an unterminated object/string is the common shape when a
    # response hits the token ceiling mid-write.
    if depth > 0 and start >= 0:
        truncated = cleaned[start:]
        for repair in ('"}' + "}" * (depth - 1), "}" * depth):
            try:
                data = json.loads(truncated + repair)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    raise ValueError("No valid JSON object in response")


def coerce(schema: Type[T], raw: str) -> T:
    """Parse ``raw`` and validate it against ``schema``.

    Raises ``ValueError`` if no JSON can be found; ``ValidationError`` if what
    was found does not fit the schema.
    """
    return schema.model_validate(salvage_json(raw))


def coerce_or_none(schema: Type[T], raw: str) -> T | None:
    """:func:`coerce`, returning None instead of raising.

    Every caller of this treats a structured response as optional enrichment —
    a failed critique, routing hint, or decomposition falls back to a rule — so
    they branch on None rather than wrapping each call in try/except.
    """
    try:
        return coerce(schema, raw)
    except (ValueError, ValidationError) as exc:
        log.info("Could not coerce response to %s: %s", schema.__name__, exc)
        return None
