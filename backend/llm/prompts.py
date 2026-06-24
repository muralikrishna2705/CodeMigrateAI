ANALYZER_PROMPT = """Analyze this {source_language} {source_version} code.
Migration target: {target_language} {target_version}

```{source_language}
{code}
```

Respond ONLY with this JSON (no other text):
{{
  "deprecated_patterns": ["list what is outdated or deprecated"],
  "migration_challenges": ["list specific things that will be hard to migrate"],
  "key_constructs": ["list main language constructs used"],
  "summary": "2 sentence plain-english summary of what this code does"
}}"""
