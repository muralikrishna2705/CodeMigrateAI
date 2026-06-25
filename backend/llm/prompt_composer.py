import hashlib
import json
from collections import OrderedDict
from typing import Any

from llm.language_profiles import LanguageProfile


class PromptCache:
    def __init__(self, maxsize: int = 100):
        self._items: OrderedDict[str, str] = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> str | None:
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def set(self, key: str, value: str):
        if key in self._items:
            self._items.pop(key)
        elif len(self._items) >= self.maxsize:
            self._items.popitem(last=False)
        self._items[key] = value

    def __len__(self) -> int:
        return len(self._items)


class PromptComposer:
    def __init__(self, max_cache_entries: int = 100):
        self._cache = PromptCache(maxsize=max_cache_entries)

    def compose(
        self,
        *,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        source_version: str,
        target_version: str,
        source_code: str,
        analyzer_context: dict[str, Any] | None,
        migration_type: str,
    ) -> str:
        analyzer_context = analyzer_context or {}
        source_code = source_code[:4000]
        cache_key = self._cache_key(
            source_profile,
            target_profile,
            source_version,
            target_version,
            source_code,
            analyzer_context,
            migration_type,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = "\n\n".join(
            [
                self._build_system_role(source_profile, target_profile, migration_type),
                self._build_language_guidance(
                    source_profile,
                    target_profile,
                    source_version,
                    target_version,
                    migration_type,
                ),
                self._build_analyzer_section(analyzer_context),
                self._build_few_shots(source_profile, target_profile, migration_type),
                self._build_source_section(source_profile, source_version, source_code),
                self._build_output_format(),
            ]
        )
        self._cache.set(cache_key, prompt)
        return prompt

    def cache_size(self) -> int:
        return len(self._cache)

    def _cache_key(
        self,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        source_version: str,
        target_version: str,
        source_code: str,
        analyzer_context: dict[str, Any],
        migration_type: str,
    ) -> str:
        context_json = json.dumps(analyzer_context, sort_keys=True, default=str)
        content = "|".join(
            [
                source_profile.language_id,
                source_version,
                target_profile.language_id,
                target_version,
                migration_type,
                hashlib.sha256(source_code.encode()).hexdigest(),
                hashlib.sha256(context_json.encode()).hexdigest(),
            ]
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _build_system_role(
        self,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        migration_type: str,
    ) -> str:
        if migration_type == "upgrade_version":
            return (
                f"You are an expert {target_profile.display_name} migration engineer. "
                f"Modernize {source_profile.display_name} code while preserving "
                "behavior."
            )
        return (
            f"You are an expert {source_profile.display_name} to "
            f"{target_profile.display_name} migration engineer. Convert code into "
            "idiomatic target-language code while preserving behavior."
        )

    def _build_language_guidance(
        self,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        source_version: str,
        target_version: str,
        migration_type: str,
    ) -> str:
        lines = [
            "LANGUAGE GUIDANCE",
            f"Source: {source_profile.display_name} {source_version}",
            f"Target: {target_profile.display_name} {target_version}",
            f"Migration type: {migration_type}",
        ]

        if migration_type == "upgrade_version":
            target_features = target_profile.get_version_features(target_version)
            if target_features:
                lines.append("Target-version features to consider:")
                lines.extend(f"- {feature}" for feature in target_features[:8])
            if target_profile.syntax_rules:
                lines.append("Modernization rules:")
                lines.extend(f"- {rule}" for rule in target_profile.syntax_rules[:8])
            if target_profile.idioms:
                lines.append("Idiomatic replacements:")
                lines.extend(
                    f"- {source} -> {target}"
                    for source, target in list(target_profile.idioms.items())[:8]
                )
            if target_profile.stdlib_mappings:
                lines.append("Standard-library migrations:")
                lines.extend(
                    f"- {source} -> {target}"
                    for source, target in list(
                        target_profile.stdlib_mappings.items()
                    )[:8]
                )
        else:
            mappings = source_profile.get_cross_language_mapping(
                target_profile.language_id
            )
            if mappings:
                lines.append("Cross-language mappings:")
                lines.extend(
                    f"- {source} -> {target}"
                    for source, target in list(mappings.items())[:12]
                )
            if target_profile.syntax_rules:
                lines.append("Target-language rules:")
                lines.extend(f"- {rule}" for rule in target_profile.syntax_rules[:8])
            if target_profile.idioms:
                lines.append("Target idioms:")
                lines.extend(
                    f"- {source} -> {target}"
                    for source, target in list(target_profile.idioms.items())[:8]
                )

        if target_profile.common_pitfalls:
            lines.append("Pitfalls to avoid:")
            lines.extend(
                f"- {pitfall}" for pitfall in target_profile.common_pitfalls[:8]
            )

        if target_profile.style_guide:
            lines.append(f"Style guide: {target_profile.style_guide}")

        return "\n".join(lines)

    def _build_analyzer_section(self, analyzer_context: dict[str, Any]) -> str:
        if not analyzer_context:
            return "ANALYZER CONTEXT\nNo analyzer findings were provided."
        return (
            "ANALYZER CONTEXT\n"
            f"{json.dumps(analyzer_context, indent=2, sort_keys=True, default=str)}"
        )

    def _build_few_shots(
        self,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        migration_type: str,
    ) -> str:
        examples = []
        for example in (
            source_profile.few_shot_examples + target_profile.few_shot_examples
        ):
            if example not in examples:
                examples.append(example)
            if len(examples) >= 5:
                break

        if not examples:
            return "FEW-SHOT EXAMPLES\nNo examples available for this pair."

        lines = ["FEW-SHOT EXAMPLES"]
        for idx, example in enumerate(examples, start=1):
            lines.extend(
                [
                    f"Example {idx}: {example.get('description', migration_type)}",
                    "Source:",
                    str(example.get("source", "")).strip(),
                    "Target:",
                    str(example.get("target", "")).strip(),
                ]
            )
        return "\n".join(lines)

    def _build_source_section(
        self, source_profile: LanguageProfile, source_version: str, source_code: str
    ) -> str:
        return (
            f"SOURCE CODE ({source_profile.display_name} {source_version})\n"
            f"```{source_profile.language_id}\n{source_code}\n```"
        )

    def _build_output_format(self) -> str:
        return (
            "OUTPUT FORMAT — CRITICAL: OUTPUT ONLY VALID JSON\n"
            "No markdown fences, no code blocks, no explanations, no prose.\n"
            "Your entire response must be a single JSON object matching this schema:\n"
            "{\n"
            '  "plan_summary": "One concise sentence summarizing the migration plan.",\n'
            '  "migrated_code": "The complete migrated source code as a string."\n'
            "}\n\n"
            "VALID EXAMPLE:\n"
            "{\"plan_summary\": \"Upgrade Java 7 to 17: use diamond inference, var, java.time\", \"migrated_code\": \"public class Foo {\\n  private List<String> list = new ArrayList<>();\\n}\"}\n\n"
            "INVALID (will cause parse failure):\n"
            "- Any text before or after the JSON\n"
            "- Markdown fences (```json ... ```)\n"
            "- Comments inside JSON\n"
            "- Missing or extra keys\n"
            "- Unescaped newlines in migrated_code (use \\n)"
        )
