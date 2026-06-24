from .base import LanguageProfile


class CppProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="cpp",
            display_name="C++",
            versions=["14", "17", "20", "23"],
            syntax_rules=[
                "Prefer smart pointers over owning raw pointers.",
                "Use range-based for loops and algorithms where readable.",
                "Use std::optional for absence and std::variant for tagged unions.",
                "Use concepts for generic constraints in C++20+ targets.",
            ],
            idioms={
                "NULL": "nullptr",
                "manual new/delete": "std::unique_ptr or automatic storage",
                "C arrays": "std::array or std::vector",
                "out params for optional values": "std::optional return values",
            },
            stdlib_mappings={
                "boost::optional": "std::optional",
                "boost::filesystem": "std::filesystem",
                "std::auto_ptr": "std::unique_ptr",
            },
            version_features={
                "17": ["std::optional", "std::variant", "structured bindings"],
                "20": ["concepts", "ranges", "coroutines"],
                "23": ["std::expected", "ranges improvements", "deducing this"],
            },
            few_shot_examples=[
                {
                    "description": "Raw null to optional",
                    "source_version": "14",
                    "target_version": "20",
                    "source": "User* findUser(int id);",
                    "target": "std::optional<User> findUser(int id);",
                }
            ],
            common_pitfalls=[
                "Do not introduce dangling references while modernizing ownership.",
                "Preserve value vs pointer semantics.",
                "Include required headers for modern standard library types.",
            ],
            style_guide=(
                "Use modern C++ standard library facilities, RAII, const-correctness, "
                "and clear ownership."
            ),
            cross_language_mappings={
                "java": {
                    "std::vector<T>": "List<T>",
                    "std::unordered_map<K,V>": "Map<K,V>",
                    "std::optional<T>": "Optional<T>",
                    "RAII": "try-with-resources or scoped ownership",
                },
                "python": {
                    "std::vector<T>": "list[T]",
                    "std::unordered_map<K,V>": "dict[K, V]",
                    "std::optional<T>": "Optional[T]",
                },
                "rust": {
                    "std::vector<T>": "Vec<T>",
                    "std::unordered_map<K,V>": "HashMap<K, V>",
                    "std::optional<T>": "Option<T>",
                    "RAII": "ownership and Drop",
                },
            },
        )
