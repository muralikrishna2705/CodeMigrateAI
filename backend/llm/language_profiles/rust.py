from .base import LanguageProfile


class RustProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="rust",
            display_name="Rust",
            versions=["1.70", "1.80"],
            syntax_rules=[
                "Use Result and Option to model failure and absence.",
                "Prefer iterators when they keep ownership clear.",
                "Use pattern matching for enum and option handling.",
                "Keep borrowing and lifetimes simple; clone only when justified.",
            ],
            idioms={
                "null": "Option<T>",
                "exceptions": "Result<T, E>",
                "manual loops": "iterators when readable",
                "mutable shared state": "Arc<Mutex<T>> or channels when needed",
            },
            stdlib_mappings={
                "hash map": "std::collections::HashMap",
                "dynamic errors": "Box<dyn std::error::Error>",
                "filesystem paths": "std::path::Path and PathBuf",
            },
            version_features={
                "1.70": ["OnceLock", "IsTerminal"],
                "1.80": [
                    "LazyLock",
                    "checked cfg names",
                    "exclusive ranges in patterns",
                ],
            },
            few_shot_examples=[
                {
                    "description": "Nullable value to Option",
                    "source_version": "1.70",
                    "target_version": "1.80",
                    "source": (
                        "let name = get_name(); // may be null in source language"
                    ),
                    "target": "let name: Option<String> = get_name();",
                }
            ],
            common_pitfalls=[
                (
                    "Do not use unwrap in migrated application logic unless "
                    "failure is impossible."
                ),
                "Preserve ownership and mutation semantics explicitly.",
                "Return Result for recoverable failures.",
            ],
            style_guide=(
                "Use rustfmt style, expressive enums, clear ownership, and standard "
                "library types before external crates."
            ),
            cross_language_mappings={
                "java": {
                    "Vec<T>": "List<T>",
                    "HashMap<K,V>": "Map<K,V>",
                    "Option<T>": "Optional<T>",
                    "Result<T,E>": "checked exception or result object",
                },
                "python": {
                    "Vec<T>": "list[T]",
                    "HashMap<K,V>": "dict[K, V]",
                    "Option<T>": "Optional[T]",
                    "Result<T,E>": "raise exceptions or return explicit result",
                },
                "go": {
                    "Vec<T>": "[]T",
                    "HashMap<K,V>": "map[K]V",
                    "Result<T,E>": "(T, error)",
                },
            },
        )
