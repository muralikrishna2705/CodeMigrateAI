from .base import LanguageProfile


class CSharpProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="csharp",
            display_name="C#",
            versions=["6", "8", "10", "12"],
            syntax_rules=[
                "Use nullable reference annotations for C# 8+ targets.",
                "Use records for immutable data where appropriate.",
                "Use pattern matching for clear type and property checks.",
                "Prefer async/await over manual task continuations.",
            ],
            idioms={
                "Tuple DTO": "record type",
                "null checks": "nullable annotations plus pattern matching",
                "Task.ContinueWith": "async/await",
                "manual properties": "auto-properties when logic is absent",
            },
            stdlib_mappings={
                "ArrayList": "List<T>",
                "Hashtable": "Dictionary<TKey, TValue>",
                "DateTime.Now": "DateTimeOffset.Now when offsets matter",
            },
            version_features={
                "8": ["nullable reference types", "switch expressions"],
                "10": ["record structs", "global usings", "file-scoped namespaces"],
                "12": ["primary constructors", "collection expressions"],
            },
            few_shot_examples=[
                {
                    "description": "DTO to record",
                    "source_version": "6",
                    "target_version": "12",
                    "source": "public class User { public string Name { get; set; } }",
                    "target": "public sealed record User(string Name);",
                }
            ],
            common_pitfalls=[
                "Do not add nullable suppression operators without justification.",
                "Keep LINQ laziness or eager evaluation behavior intact.",
                "Preserve async exception behavior.",
            ],
            style_guide=(
                "Use .NET naming conventions, clear nullable intent, and concise "
                "records or pattern matching where they improve readability."
            ),
            cross_language_mappings={
                "java": {
                    "List<T>": "List<T>",
                    "Dictionary<TKey,TValue>": "Map<K,V>",
                    "Console.WriteLine": "System.out.println",
                    "async Task": "CompletableFuture or structured async equivalent",
                },
                "python": {
                    "List<T>": "list[T]",
                    "Dictionary<TKey,TValue>": "dict[K, V]",
                    "record": "dataclass",
                    "null": "None",
                },
                "typescript": {
                    "record/class DTO": "interface or type alias",
                    "List<T>": "T[]",
                    "Dictionary<string,T>": "Record<string, T>",
                },
            },
        )
