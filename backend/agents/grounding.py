"""Post-hoc grounding check for migrated code.

Flags library imports in the migrated output that are grounded by nothing: not
present in the source, not shown in the retrieved reference context, and not part
of the target language's standard library. Invented third-party dependencies are
the highest-signal, lowest-false-positive hallucination to catch — a legitimate
migration imports target stdlib modules or the libraries demonstrated in the
reference examples, so an import matching none of those is a strong "the model
made this up" signal.

Matching is by *namespace segment*, not raw substring. ``imp in source_code``
matched anywhere in the text, so an import was "grounded" by its letters
appearing inside an unrelated word — ``io.netty.channel`` counted as verified
because ``io`` occurs inside ``java.io.File`` in the profile's mapping table.
:func:`_mentions` accepts a hit only when the surrounding characters end a
segment, which is what makes the answer mean what it claims.

The search itself stays a C-level ``str.find`` sweep rather than a prefix index.
A trie answers each lookup in O(L) against the scan's O(N), but building one
costs a Python-level pass over every character of the context — measured at
~15x the total cost here, because a migration checks ~10 imports against a
context of tens of KB, and ten lookups never amortize the build. The index would
start paying only in the hundreds of imports.

This is advisory: it is surfaced in the MigratorAgent report (details["grounding"])
for visibility and does not fail the migration.
"""

import re

from llm.language_profiles import LanguageProfile, ProfileRegistry

# How to pull import module/package names from a code string, per language.
_IMPORT_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE),
        re.compile(r"^\s*from\s+([\w.]+)\s+import\b", re.MULTILINE),
    ],
    "java": [re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)],
    "kotlin": [re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE)],
    "csharp": [re.compile(r"^\s*using\s+(?:static\s+)?([\w.]+)\s*;", re.MULTILINE)],
    "javascript": [
        re.compile(r"""import\s+[^;]*?from\s+['"]([^'"]+)['"]"""),
        re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    "typescript": [
        re.compile(r"""import\s+[^;]*?from\s+['"]([^'"]+)['"]"""),
        re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
    ],
    "rust": [re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE)],
    "cpp": [re.compile(r'^\s*#include\s*[<"]([\w./]+)[>"]', re.MULTILINE)],
    # Go's import block is handled specially in _extract_imports.
    "go": [],
}

# Standard-library roots that are always considered grounded. These are the
# top-level package/namespace roots, not exhaustive module lists.
_STDLIB_ROOTS: dict[str, set[str]] = {
    "python": {
        "os", "sys", "re", "json", "math", "collections", "itertools",
        "functools", "typing", "datetime", "time", "pathlib", "asyncio", "abc",
        "dataclasses", "enum", "logging", "random", "io", "subprocess",
        "threading", "multiprocessing", "unittest", "hashlib", "http", "urllib",
        "socket", "struct", "copy", "string", "decimal", "fractions",
        "statistics", "contextlib", "warnings", "traceback", "inspect",
        "importlib", "argparse", "csv", "sqlite3", "xml", "html", "email",
        "uuid", "base64", "secrets", "shutil", "tempfile", "glob", "pickle",
        "queue", "heapq", "bisect", "array", "weakref", "gc", "operator",
        "textwrap", "difflib", "signal", "select", "ctypes", "platform",
        "getpass", "shlex", "zipfile", "tarfile", "gzip", "configparser",
    },
    "java": {"java", "javax", "jakarta"},
    "kotlin": {"kotlin", "kotlinx", "java", "javax"},
    "csharp": {"System", "Microsoft"},
    # Node built-ins are grounded; anything else bare is an npm package.
    "javascript": {
        "fs", "path", "http", "https", "os", "url", "util", "events", "stream",
        "crypto", "buffer", "process", "child_process", "assert", "net", "dns",
        "zlib", "querystring", "readline", "tls", "cluster", "worker_threads",
    },
    "typescript": {
        "fs", "path", "http", "https", "os", "url", "util", "events", "stream",
        "crypto", "buffer", "process", "child_process", "assert", "net", "dns",
        "zlib", "querystring", "readline", "tls", "cluster", "worker_threads",
    },
    "rust": {"std", "core", "alloc", "crate", "self", "super"},
    # Top-level roots of the Go standard library. An import is stdlib iff its
    # first path segment is one of these (so "net/http", "encoding/json" resolve
    # via "net"/"encoding"). This replaces the old "any dot-free import is
    # stdlib" heuristic, which wrongly grounded invented packages like
    # "fastjson". A dotted first segment is a module host (github.com/...) and
    # never matches, because "." does not end a segment in Go's separator set.
    "go": {
        "archive", "bufio", "builtin", "bytes", "cmp", "compress", "container",
        "context", "crypto", "database", "debug", "embed", "encoding", "errors",
        "expvar", "flag", "fmt", "go", "hash", "html", "image", "index", "io",
        "iter", "log", "maps", "math", "mime", "net", "os", "path", "plugin",
        "reflect", "regexp", "runtime", "slices", "sort", "strconv", "strings",
        "sync", "syscall", "testing", "text", "time", "unicode", "unsafe",
    },
    "cpp": set(),  # handled via the header-shape rule in _is_grounded
}

# What ends a namespace segment, per language. This is per-language because the
# rules genuinely differ: Go's "." is *not* a separator, which is precisely why
# "image.example.com/x" must not resolve to the stdlib root "image" while
# "image/png" must.
_SEPARATORS: dict[str, str] = {
    "python": ".",
    "java": ".",
    "kotlin": ".",
    "csharp": ".",
    "javascript": "/",
    "typescript": "/",
    "rust": ":",
    "go": "/",
    "cpp": "/.",
}
_DEFAULT_SEPARATORS = "./:"

# Characters that continue an identifier. A candidate hit flanked by one of
# these is part of a longer name, not the namespace we asked about.
_WORD = re.compile(r"[A-Za-z0-9_]")


def _separators(lang: str) -> str:
    return _SEPARATORS.get(lang, _DEFAULT_SEPARATORS)


def _mentions(text: str, term: str, separators: str) -> bool:
    """True when ``text`` names ``term`` as a whole namespace segment.

    Accepts the exact name and any namespace nested under it — ``requests``
    matches a context using ``requests.get`` — while rejecting the substring
    hits a plain ``in`` test would take: ``requests`` inside ``requestshandler``
    (trailing word character) and ``mycorp`` inside ``com.mycorp`` (leading
    separator, so the term is not the root).

    ``str.find`` does the scanning in C and the boundary test only runs on the
    rare candidate hit, which is what keeps this cheaper than indexing the text.
    """
    if not text or not term:
        return False
    start = 0
    end_of_term = len(term)
    while True:
        index = text.find(term, start)
        if index < 0:
            return False
        before = text[index - 1] if index else ""
        after_index = index + end_of_term
        after = text[after_index] if after_index < len(text) else ""
        if not (before and (_WORD.match(before) or before in separators)) and not (
            after and _WORD.match(after)
        ):
            return True
        start = index + 1


def _evidence_texts(
    source_code: str, rag_context: str, profile: LanguageProfile | None
) -> list[str]:
    """The texts an import may be grounded in, assembled once per check.

    The profile's mapping table used to be re-joined inside the per-import loop,
    so a file with 12 imports rebuilt the same string 12 times.
    """
    texts = [text for text in (source_code, rag_context) if text]
    if profile is not None:
        texts.append(
            " ".join(
                list(profile.stdlib_mappings.values())
                + list(profile.stdlib_mappings.keys())
                + list(profile.idioms.values())
            )
        )
    return texts


def _extract_imports(code: str, language: str) -> list[str]:
    lang = ProfileRegistry.normalize(language)
    if lang == "go":
        return _extract_go_imports(code)
    imports: list[str] = []
    for pattern in _IMPORT_PATTERNS.get(lang, []):
        imports.extend(pattern.findall(code))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique = []
    for imp in imports:
        if imp and imp not in seen:
            seen.add(imp)
            unique.append(imp)
    return unique


def _extract_go_imports(code: str) -> list[str]:
    imports: list[str] = []
    # Block form: import ( "a" \n "b/c" )
    for block in re.findall(r"import\s*\(([\s\S]*?)\)", code):
        imports.extend(re.findall(r'"([^"]+)"', block))
    # Single form: import "a"
    imports.extend(re.findall(r'^\s*import\s+"([^"]+)"', code, re.MULTILINE))
    seen: set[str] = set()
    return [i for i in imports if not (i in seen or seen.add(i))]


def _root(imp: str, language: str) -> str:
    lang = ProfileRegistry.normalize(language)
    if lang == "rust":
        return imp.split("::", 1)[0]
    if lang == "go":
        # Stdlib packages are single-segment or slash-paths with no dotted domain
        # (e.g. "net/http"); third-party start with a host like "github.com/...".
        return imp.split("/", 1)[0]
    if lang in ("javascript", "typescript"):
        # Scoped packages: @scope/name -> @scope/name; plain: pkg/sub -> pkg.
        if imp.startswith("@"):
            return "/".join(imp.split("/", 2)[:2])
        return imp.split("/", 1)[0]
    # Dotted namespaces (python/java/kotlin/csharp).
    return imp.split(".", 1)[0]


def _is_relative_js(imp: str) -> bool:
    return imp.startswith(".") or imp.startswith("/")


def _is_grounded(imp: str, language: str, evidence: list[str]) -> bool:
    lang = ProfileRegistry.normalize(language)
    separators = _separators(lang)

    # Local / relative imports are never third-party inventions.
    if lang in ("javascript", "typescript") and _is_relative_js(imp):
        return True

    # Standard library. Every root here is a single segment, so comparing the
    # import's root against the set answers it directly — and because Go's root
    # keeps its dotted host ("github.com"), a module path can never collide with
    # a stdlib name.
    root = _root(imp, language)
    if root in _STDLIB_ROOTS.get(lang, ()):
        return True

    # Carried over from the source, demonstrated in the retrieved examples, or
    # named in the target profile's curated mappings. Both the full path and its
    # root count, so a context showing "requests.get" grounds "import requests"
    # and one showing "com.mycorp.util" grounds a child of that package.
    for text in evidence:
        if _mentions(text, imp, separators) or _mentions(text, root, separators):
            return True

    if lang == "cpp":
        # A header with no directory and no extension is a stdlib header
        # (<vector>); anything else needed evidence, checked above.
        return "/" not in imp and "." not in imp

    return False


def check_import_grounding(
    migrated_code: str,
    target_language: str,
    source_code: str = "",
    rag_context: str = "",
) -> dict:
    """Return a grounding report for the migrated code's imports.

    Shape: ``{"checked": bool, "total_imports": int, "unverified_imports": [...]}``.
    ``checked`` is False when the target language has no import extractor, so
    callers can distinguish "nothing to flag" from "not analyzed".
    """
    lang = ProfileRegistry.normalize(target_language)
    if lang not in _IMPORT_PATTERNS:
        return {"checked": False, "total_imports": 0, "unverified_imports": []}

    imports = _extract_imports(migrated_code, target_language)
    if not imports:
        return {"checked": True, "total_imports": 0, "unverified_imports": []}

    # Only assemble the evidence texts once there is something to look up in them.
    profile = ProfileRegistry.get(target_language)
    evidence = _evidence_texts(source_code, rag_context, profile)
    unverified = [
        imp for imp in imports if not _is_grounded(imp, target_language, evidence)
    ]
    return {
        "checked": True,
        "total_imports": len(imports),
        "unverified_imports": unverified,
    }
