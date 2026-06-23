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

PLANNER_PROMPT = """Create a migration plan:
  Source: {source_language} {source_version}
  Target: {target_language} {target_version}
  Type: {migration_type}
{recipe_context}
{analysis_context}
Respond ONLY with this JSON (no other text):
{{
  "strategy": "one sentence describing the overall approach",
  "syntax_changes": ["specific syntax transformation 1", "..."],
  "api_changes": ["API mapping 1", "..."],
  "risk_areas": ["thing that needs careful handling 1", "..."],
  "estimated_effort": "low | medium | high"
}}"""
