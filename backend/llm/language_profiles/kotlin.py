from .base import LanguageProfile


class KotlinProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="kotlin",
            display_name="Kotlin",
            versions=["1.7", "1.9", "2.0"],
            syntax_rules=[
                "Use nullable types instead of sentinel null conventions.",
                "Prefer data classes for value objects.",
                "Use collection operators for simple transformations.",
                "Use coroutines for asynchronous flows when equivalent.",
            ],
            idioms={
                "getters/setters": "properties",
                "POJO": "data class",
                "null checks": "safe calls, Elvis operator, and nullable types",
                "builder callbacks": "DSL-style lambdas when clear",
            },
            stdlib_mappings={
                "java.util.Optional": "nullable type",
                "java.util.List": "List or MutableList",
                "java.util.Map": "Map or MutableMap",
            },
            version_features={
                "1.9": ["data object", "enum entries", "K2 preview"],
                "2.0": ["K2 compiler", "improved type inference"],
            },
            few_shot_examples=[
                {
                    "description": "Java-style DTO to Kotlin data class",
                    "source_version": "1.7",
                    "target_version": "2.0",
                    "source": "class User(val id: Int, val name: String)",
                    "target": "data class User(val id: Int, val name: String)",
                }
            ],
            common_pitfalls=[
                "Do not use platform types without null-safety consideration.",
                "Preserve mutability intent between List and MutableList.",
                "Keep coroutine scopes explicit.",
            ],
            style_guide=(
                "Use idiomatic Kotlin properties, expression bodies for short "
                "functions, and null-safe APIs."
            ),
            cross_language_mappings={
                "java": {
                    "data class": "record or class",
                    "List<T>": "List<T>",
                    "T?": "Optional<T> or nullable reference",
                    "coroutine": "CompletableFuture or structured async equivalent",
                },
                "python": {
                    "data class": "dataclass",
                    "List<T>": "list[T]",
                    "Map<K,V>": "dict[K, V]",
                    "T?": "Optional[T]",
                },
                "typescript": {
                    "data class": "interface or type alias",
                    "List<T>": "T[]",
                    "T?": "T | null",
                },
            },
        )
