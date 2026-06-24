from .base import LanguageProfile


class TypeScriptProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="typescript",
            display_name="TypeScript",
            versions=["3.x", "4.x", "5.x"],
            syntax_rules=[
                "Add explicit parameter and return types where inference is weak.",
                "Prefer unknown over any for untrusted values.",
                "Represent object shapes with interfaces or type aliases.",
                "Use discriminated unions for tagged variants.",
                "Keep strict null checks satisfied.",
            ],
            idioms={
                "any": "unknown plus narrowing when possible",
                "string constants": "literal union types",
                "callback params": "typed function signatures",
                "plain object contracts": "interface or type alias",
            },
            stdlib_mappings={
                "require": "import",
                "module.exports": "export default or named exports",
                "Object": "Record<string, unknown> when shape is dynamic",
            },
            version_features={
                "4.x": ["template literal types", "satisfies-like narrowing patterns"],
                "5.x": [
                    "satisfies operator",
                    "const type parameters",
                    "decorator updates",
                ],
            },
            few_shot_examples=[
                {
                    "description": "JavaScript function to typed TypeScript",
                    "source_version": "ES2020",
                    "target_version": "5.x",
                    "source": "function total(items) { return items.length; }",
                    "target": (
                        "function total<T>(items: T[]): number { "
                        "return items.length; }"
                    ),
                },
                {
                    "description": "Object shape extraction",
                    "source_version": "ES2020",
                    "target_version": "5.x",
                    "source": "const user = { id: 1, name: 'Ada' };",
                    "target": (
                        "interface User { id: number; name: string; }\n"
                        "const user: User = { id: 1, name: 'Ada' };"
                    ),
                },
            ],
            common_pitfalls=[
                "Do not silence errors with broad any casts.",
                "Preserve runtime behavior while adding types.",
                "Narrow unknown values before property access.",
            ],
            style_guide=(
                "Use strict, readable types. Prefer interfaces for public object "
                "contracts and type aliases for unions or mapped types."
            ),
            cross_language_mappings={
                "javascript": {
                    "interfaces/types": "runtime objects only",
                    "readonly": "Object.freeze or conventions if needed",
                    "enums": "objects or literal strings",
                },
                "python": {
                    "interface": "Protocol, dataclass, or TypedDict",
                    "Promise<T>": "async def returning T",
                    "T | undefined": "Optional[T]",
                },
                "java": {
                    "interface": "interface or record",
                    "T[]": "List<T>",
                    "Promise<T>": "CompletableFuture<T>",
                },
            },
        )
