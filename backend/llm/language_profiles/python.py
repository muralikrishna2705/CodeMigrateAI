from .base import LanguageProfile


class PythonProfile(LanguageProfile):
    def __init__(self):
        super().__init__(
            language_id="python",
            display_name="Python",
            versions=["2.7", "3.8", "3.10", "3.12"],
            syntax_rules=[
                "Use print() and Python 3 iterator semantics.",
                "Prefer f-strings for string interpolation.",
                "Use pathlib for filesystem paths.",
                "Use dataclasses for simple structured data.",
                "Use match statements only when they clarify branching.",
            ],
            idioms={
                "% formatting": "f-strings",
                "urllib2": "urllib.request",
                "dict.keys() as list": "list(dict.keys())",
                "plain data dict": "@dataclass when fields are stable",
            },
            stdlib_mappings={
                "urllib2": "urllib.request",
                "ConfigParser": "configparser",
                "cPickle": "pickle",
                "StringIO.StringIO": "io.StringIO",
            },
            version_features={
                "3.8": ["assignment expressions", "positional-only parameters"],
                "3.10": ["match statements", "union types"],
                "3.12": ["faster CPython", "improved typing", "pathlib improvements"],
            },
            few_shot_examples=[
                {
                    "description": "Python 2 print and formatting to Python 3",
                    "source_version": "2.7",
                    "target_version": "3.12",
                    "source": 'print "User: %s" % name',
                    "target": 'print(f"User: {name}")',
                },
                {
                    "description": "urllib2 to urllib.request",
                    "source_version": "2.7",
                    "target_version": "3.12",
                    "source": "resp = urllib2.urlopen(url)",
                    "target": "resp = urllib.request.urlopen(url)",
                },
            ],
            common_pitfalls=[
                "Do not preserve Python 2 bytes/text ambiguity.",
                "Do not use mutable defaults in function signatures.",
                "Keep indentation and significant whitespace valid.",
                "Preserve exception behavior and resource cleanup.",
            ],
            style_guide=(
                "Follow PEP 8 naming, use type hints when useful, and keep code "
                "simple, readable, and explicit about None handling."
            ),
            cross_language_mappings={
                "java": {
                    "dict": "Map<K, V>",
                    "list": "List<T>",
                    "None": "null or Optional<T>",
                    "with": "try-with-resources",
                },
                "javascript": {
                    "dict": "object literal or Map",
                    "list": "Array",
                    "None": "null",
                    "async def": "async function",
                },
                "typescript": {
                    "dict": "Record<string, T> or Map<K, V>",
                    "list": "T[]",
                    "Optional": "T | undefined",
                    "dataclass": "interface or class",
                },
                "go": {
                    "dict": "map[K]V",
                    "list": "[]T",
                    "None": "nil plus explicit error/value handling",
                },
                "rust": {
                    "dict": "HashMap<K, V>",
                    "list": "Vec<T>",
                    "None": "Option::None",
                    "exceptions": "Result<T, E>",
                },
            },
        )
