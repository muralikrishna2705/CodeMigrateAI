"""Post-hoc grounding check for migrated code.

Flags library imports in the migrated output that are grounded by nothing: not
present in the source, not shown in the retrieved reference context, and not part
of the target language's standard library. Invented third-party dependencies are
the highest-signal, lowest-false-positive hallucination to catch — a legitimate
migration imports target stdlib modules or the libraries demonstrated in the
reference examples, so an import matching none of those is a strong "the model
made this up" signal.

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
    "cpp": set(),  # handled via _is_cpp_stdlib
    "go": set(),   # handled via _is_go_stdlib
}


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


def _is_go_stdlib(imp: str) -> bool:
    # Third-party Go imports carry a dotted host in the first path segment.
    first = imp.split("/", 1)[0]
    return "." not in first


def _is_relative_js(imp: str) -> bool:
    return imp.startswith(".") or imp.startswith("/")


def _is_grounded(
    imp: str,
    language: str,
    source_code: str,
    rag_context: str,
    profile: LanguageProfile | None,
) -> bool:
    lang = ProfileRegistry.normalize(language)
    root = _root(imp, language)

    # Local / relative imports are never third-party inventions.
    if lang in ("javascript", "typescript") and _is_relative_js(imp):
        return True
    if lang == "go" and _is_go_stdlib(imp):
        return True
    if lang == "cpp":
        # Header with no directory + no extension is a stdlib header (<vector>);
        # anything the source already included is grounded too.
        if imp in source_code or imp in rag_context:
            return True
        return "/" not in imp and "." not in imp

    # Standard-library roots are always grounded.
    stdlib = _STDLIB_ROOTS.get(lang, set())
    # C#/Java roots are case-sensitive (System); others compare as-is.
    if root in stdlib:
        return True

    # Carried over from the source, or demonstrated in the retrieved examples.
    if imp in source_code or root in source_code:
        return True
    if rag_context and (imp in rag_context or root in rag_context):
        return True

    # Present in the target profile's curated stdlib/idiom mappings.
    if profile is not None:
        mapping_text = " ".join(
            list(profile.stdlib_mappings.values())
            + list(profile.stdlib_mappings.keys())
            + list(profile.idioms.values())
        )
        if imp in mapping_text or root in mapping_text:
            return True

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

    profile = ProfileRegistry.get(target_language)
    imports = _extract_imports(migrated_code, target_language)
    unverified = [
        imp
        for imp in imports
        if not _is_grounded(imp, target_language, source_code, rag_context, profile)
    ]
    return {
        "checked": True,
        "total_imports": len(imports),
        "unverified_imports": unverified,
    }
