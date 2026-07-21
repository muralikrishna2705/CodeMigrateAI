import hashlib
import re

from models.state import MigrationState

KEY_NAMESPACE = "migrate"
# Components go into a ":"-delimited key, so any ":" inside one would forge a
# level boundary and let two different migrations collide on a prefix query.
_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]")


def normalize_source(source: str) -> str:
    """Fold away source edits that cannot change the migration.

    Hashing raw bytes makes the cache miss on differences no migration would
    ever reflect. The common one is not a code change at all: the same file
    checked out on Windows and on Linux differs in every line ending, so a
    CRLF/LF flip misses the entire cache.

    Deliberately conservative. It normalizes line endings, strips trailing
    whitespace, and trims leading/trailing blank lines — edits that are
    invisible to a parser *and* to the model reading the file.

    It specifically does **not** strip comments, which was the obvious next
    step and is wrong: the migrator carries comments through to the output, so
    two sources differing only in their comments have genuinely different
    correct migrations. Treating them as one cache entry would hand the second
    caller the first caller's comments — a wrong answer, silently, with no
    error anywhere. Nor does it touch indentation, which is syntax in Python.
    """
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _segment(value: str) -> str:
    """Normalize one key component into a single, separator-safe level."""
    return _UNSAFE.sub("_", (value or "any").strip().lower()) or "any"


def key_prefix(
    source_language: str = "",
    source_version: str = "",
    target_language: str = "",
    target_version: str = "",
) -> str:
    """Build the key prefix naming a migration family, most general first.

    Supplying a subset yields the prefix for everything beneath it, so
    ``key_prefix("python")`` covers every migration out of Python regardless of
    version or target. Stops at the first omitted component — the levels are
    ordered, so there is no way to pin a target while leaving the source open.
    """
    parts = [KEY_NAMESPACE]
    for value in (source_language, source_version, target_language, target_version):
        if not value:
            break
        parts.append(_segment(value))
    return ":".join(parts) + ":"


def generate_key(state: MigrationState) -> str:
    """Hierarchical cache key: ``migrate:<src>:<srcver>:<tgt>:<tgtver>:<hash>``.

    The language pair leads and the content digest trails, so the key is a path
    rather than an opaque blob. That ordering is what makes prefix invalidation
    possible: when a language profile or the RAG corpus for a target changes,
    every cached migration into that target is stale, and the ordering lets both
    the local trie and Redis ``SCAN MATCH`` find exactly those entries. A flat
    ``migrate:<hash>`` forces the alternative — drop the whole cache, including
    the unaffected majority.

    The digest is taken over :func:`normalize_source` rather than the raw text,
    so formatting-only differences (line endings above all) hit the same entry.
    """
    content = (
        f"{normalize_source(state.source_code)}|"
        f"{state.source_language}|{state.source_version}|"
        f"{state.target_language}|{state.target_version}"
    )
    # Every level is emitted (missing ones as "any") so the digest always sits at
    # the same depth — a variable-depth key would let one migration's prefix
    # swallow another's full key.
    levels = [
        _segment(state.source_language),
        _segment(state.source_version),
        _segment(state.target_language),
        _segment(state.target_version),
    ]
    digest = hashlib.sha256(content.encode()).hexdigest()
    return ":".join([KEY_NAMESPACE, *levels, digest])
