from .base import LanguageProfile


class GoProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="go",
            display_name="Go",
            versions=["1.18", "1.20", "1.22"],
            syntax_rules=[
                "Use generics for reusable typed containers or helpers.",
                "Prefer errors.Is and errors.As for wrapped errors.",
                "Use context.Context for cancellable operations.",
                "Keep goroutine ownership and channel closing explicit.",
            ],
            idioms={
                "interface{}": "any or a type parameter when appropriate",
                "manual error strings": "fmt.Errorf with %w for wrapping",
                "shared mutable state": "sync primitives or channels",
            },
            stdlib_mappings={
                "ioutil": "io and os package replacements",
                "errors.New wrapping": "fmt.Errorf(\"...: %w\", err)",
            },
            version_features={
                "1.18": ["generics", "fuzzing"],
                "1.20": ["errors.Join", "comparable improvements"],
                "1.22": ["range loop variable fixes", "ServeMux pattern improvements"],
            },
            few_shot_examples=[
                {
                    "description": "ioutil replacement",
                    "source_version": "1.17",
                    "target_version": "1.22",
                    "source": "data, err := ioutil.ReadFile(path)",
                    "target": "data, err := os.ReadFile(path)",
                }
            ],
            common_pitfalls=[
                "Always check and return errors deliberately.",
                "Do not hide data races when converting concurrent code.",
                "Keep package names short and idiomatic.",
            ],
            style_guide=(
                "Use gofmt-compatible formatting, small functions, explicit error "
                "paths, and simple standard library APIs."
            ),
            cross_language_mappings={
                "java": {
                    "struct": "class or record",
                    "map[K]V": "Map<K,V>",
                    "[]T": "List<T>",
                    "error return": "exception or result object",
                },
                "python": {
                    "struct": "dataclass",
                    "map": "dict",
                    "slice": "list",
                    "error return": "raise or return explicit tuple",
                },
                "rust": {
                    "error return": "Result<T, E>",
                    "slice": "Vec<T>",
                    "map": "HashMap<K, V>",
                },
            },
        )
