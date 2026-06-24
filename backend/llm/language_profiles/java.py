from .base import LanguageProfile


class JavaProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="java",
            display_name="Java",
            versions=["7", "8", "11", "17", "21"],
            syntax_rules=[
                "Use diamond inference for generic constructors.",
                "Use lambdas and method references for simple anonymous classes.",
                "Prefer switch expressions where supported by the target version.",
                "Use records for transparent immutable data carriers when appropriate.",
                "Use var only when it improves local readability.",
            ],
            idioms={
                "Collections.sort(list)": "list.sort(null)",
                "new Date()": "LocalDateTime.now()",
                "StringBuffer": "StringBuilder for non-synchronized builders",
                "anonymous Runnable": "lambda expression",
            },
            stdlib_mappings={
                "java.util.Date": "java.time.LocalDateTime",
                "java.util.Calendar": "java.time.LocalDate",
                "java.io.File": "java.nio.file.Path",
                "javax.*": "jakarta.* for Jakarta EE 9+ targets",
            },
            version_features={
                "8": ["lambdas", "streams", "Optional", "java.time"],
                "11": ["var in lambda parameters", "new String APIs", "HttpClient"],
                "17": ["records", "sealed classes", "pattern matching for instanceof"],
                "21": ["virtual threads", "record patterns", "sequenced collections"],
            },
            few_shot_examples=[
                {
                    "description": "Java 7 generic construction to Java 17",
                    "source_version": "7",
                    "target_version": "17",
                    "source": "List<String> names = new ArrayList<String>();",
                    "target": "var names = new ArrayList<String>();",
                },
                {
                    "description": "Java Date modernization",
                    "source_version": "7",
                    "target_version": "17",
                    "source": "Date createdAt = new Date();",
                    "target": "LocalDateTime createdAt = LocalDateTime.now();",
                },
            ],
            common_pitfalls=[
                "Do not change public behavior while modernizing syntax.",
                "Do not replace checked exception handling with ignored failures.",
                "Avoid raw types and unchecked casts.",
                "Add required imports for java.time, streams, and collections.",
            ],
            style_guide=(
                "Keep Java naming conventions, explicit access modifiers, and clear "
                "generic types. Prefer standard library APIs over helper code."
            ),
            cross_language_mappings={
                "python": {
                    "ArrayList": "list",
                    "HashMap": "dict",
                    "Optional": "typing.Optional or None",
                    "System.out.println": "print",
                    "java.io.File": "pathlib.Path",
                    "Stream.map/filter": "list comprehensions or generator expressions",
                },
                "javascript": {
                    "ArrayList": "Array",
                    "HashMap": "Map or object literal",
                    "Optional": "undefined/null checks",
                    "System.out.println": "console.log",
                },
                "typescript": {
                    "ArrayList<T>": "T[]",
                    "HashMap<K,V>": "Map<K, V> or Record<string, V>",
                    "Optional<T>": "T | undefined",
                    "interfaces/classes": "typed interfaces and classes",
                },
                "csharp": {
                    "ArrayList/List": "List<T>",
                    "HashMap": "Dictionary<TKey, TValue>",
                    "Optional": "nullable reference/value types",
                    "System.out.println": "Console.WriteLine",
                },
                "go": {
                    "class": "struct plus methods",
                    "ArrayList": "slice",
                    "HashMap": "map[K]V",
                    "exceptions": "error return values",
                },
                "rust": {
                    "class": "struct plus impl",
                    "ArrayList": "Vec<T>",
                    "HashMap": "std::collections::HashMap",
                    "Optional": "Option<T>",
                },
                "kotlin": {
                    "getters/setters": "properties",
                    "Optional": "nullable types",
                    "ArrayList": "MutableList<T>",
                    "HashMap": "MutableMap<K, V>",
                },
                "cpp": {
                    "ArrayList": "std::vector",
                    "HashMap": "std::unordered_map",
                    "Optional": "std::optional",
                    "String": "std::string",
                },
            },
        )
