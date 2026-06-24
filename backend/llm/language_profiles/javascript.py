from .base import LanguageProfile


class JavaScriptProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="javascript",
            display_name="JavaScript",
            versions=["ES5", "ES6", "ES2020", "ES2022"],
            syntax_rules=[
                "Replace var with const or let according to reassignment.",
                "Prefer arrow functions for concise callbacks.",
                "Use async/await for promise chains when behavior is unchanged.",
                "Use optional chaining and nullish coalescing where safe.",
                "Prefer modules over CommonJS for modern targets.",
            ],
            idioms={
                "function callbacks": "arrow functions",
                "Promise.then chains": "async/await",
                "arguments": "rest parameters",
                "module.exports": "export default or named exports",
            },
            stdlib_mappings={
                "Object.assign": "object spread for simple merges",
                "Array.prototype.indexOf": "includes for presence checks",
                "require": "import",
            },
            version_features={
                "ES6": ["classes", "let/const", "arrow functions", "modules"],
                "ES2020": [
                    "optional chaining",
                    "nullish coalescing",
                    "Promise.allSettled",
                ],
                "ES2022": ["class fields", "top-level await", "at()"],
            },
            few_shot_examples=[
                {
                    "description": "ES5 callback to modern JavaScript",
                    "source_version": "ES5",
                    "target_version": "ES2022",
                    "source": "items.map(function (item) { return item.name; });",
                    "target": "items.map((item) => item.name);",
                },
                {
                    "description": "CommonJS to module export",
                    "source_version": "ES5",
                    "target_version": "ES2022",
                    "source": "module.exports = service;",
                    "target": "export default service;",
                },
            ],
            common_pitfalls=[
                "Do not change this binding when converting functions to arrows.",
                "Keep asynchronous ordering and error handling intact.",
                "Do not introduce browser APIs in Node-only code unless requested.",
            ],
            style_guide=(
                "Use clear const/let choices, modern module syntax for modern "
                "targets, and concise functions without obscuring side effects."
            ),
            cross_language_mappings={
                "typescript": {
                    "object shapes": "interfaces or type aliases",
                    "arrays": "typed arrays such as Item[]",
                    "callbacks": "typed function signatures",
                    "null/undefined": "strict nullable unions",
                },
                "python": {
                    "Array": "list",
                    "object": "dict or dataclass",
                    "Promise": "asyncio awaitable when async behavior is needed",
                    "console.log": "print",
                },
                "java": {
                    "Array": "List<T>",
                    "object": "Map<String, Object> or a class",
                    "Promise": "CompletableFuture",
                },
            },
        )
