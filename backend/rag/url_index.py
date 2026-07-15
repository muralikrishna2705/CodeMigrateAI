"""Per-language registry of official documentation URLs (RAG Source 3)."""

# Version-agnostic official docs (tutorials / library references). These are the
# same URL regardless of target version, so every chunk fetched from them is
# tagged with the wildcard version ("any") and still matches the version-filtered
# retrieval leg — they ground *language* usage, not *version* differences.
OFFICIAL_DOC_URLS: dict[str, list[str]] = {
    "python": [
        "https://docs.python.org/3/tutorial/",
        "https://docs.python.org/3/library/",
    ],
    "java": [
        "https://docs.oracle.com/javase/tutorial/",
        "https://docs.oracle.com/en/java/javase/",
    ],
    "javascript": [
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/",
    ],
    "typescript": [
        "https://www.typescriptlang.org/docs/handbook/",
    ],
    "csharp": [
        "https://learn.microsoft.com/en-us/dotnet/csharp/",
    ],
    "go": [
        "https://go.dev/doc/tutorial/",
    ],
    "kotlin": [
        "https://kotlinlang.org/docs/home.html",
    ],
    "rust": [
        "https://doc.rust-lang.org/book/",
        "https://doc.rust-lang.org/std/",
    ],
    "cpp": [
        "https://en.cppreference.com/w/",
    ],
}


# The doc_type folder these versioned docs are fetched into. It is one of
# ``rag_migration_doc_types`` on purpose: "What's New" / release-notes pages are
# the authoritative record of what a version added and how to port to it, so they
# earn both the exact-version boost and the migration-doc boost during ranking.
VERSIONED_DOC_TYPE = "release-notes"

# Per-version official docs (RAG Source 3, version-aware). Unlike
# OFFICIAL_DOC_URLS, these pages genuinely differ per version — the "What's New",
# release-notes, and migration guides. Fetching them into
# ``_fetched/<version>/<VERSIONED_DOC_TYPE>/`` stamps each chunk with a real
# ``version`` + authoritative ``doc_type``, which is what activates the
# version-aware retrieval ladder and the metadata ranking boosts.
#
# INVARIANT: every version key MUST match a value in that language's
# ``supported_languages[...]["versions"]`` in config, otherwise the retrieval
# filter (keyed on the user-selected target_version) can never match these docs.
# ``tests/test_rag.py::TestVersionedDocRegistry`` enforces this.
VERSIONED_DOC_URLS: dict[str, dict[str, list[str]]] = {
    "python": {
        "3.12": ["https://docs.python.org/3/whatsnew/3.12.html"],
        "3.10": ["https://docs.python.org/3/whatsnew/3.10.html"],
        "3.8": ["https://docs.python.org/3/whatsnew/3.8.html"],
    },
    "csharp": {
        "12": ["https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-12"],
        "10": ["https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-10"],
        "8": ["https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-8"],
    },
    "java": {
        "21": [
            "https://docs.oracle.com/en/java/javase/21/migrate/significant-changes-jdk-release.html"
        ],
        "17": [
            "https://docs.oracle.com/en/java/javase/17/migrate/significant-changes-jdk-release.html"
        ],
    },
    "typescript": {
        "5.x": [
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-5-0.html"
        ],
        "4.x": [
            "https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-9.html"
        ],
    },
    "go": {
        "1.22": ["https://go.dev/doc/go1.22"],
        "1.20": ["https://go.dev/doc/go1.20"],
        "1.18": ["https://go.dev/doc/go1.18"],
    },
}
