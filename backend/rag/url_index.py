"""Per-language registry of official documentation URLs (RAG Source 3)."""

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
