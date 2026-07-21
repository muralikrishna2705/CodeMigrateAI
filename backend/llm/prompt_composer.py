import hashlib
import json
from typing import Any

from config import get_settings
from dsa import PrefixLRU
from llm.language_profiles import LanguageProfile


class PromptComposer:
    """Assembles the migration prompt, cached at two granularities.

    The whole-prompt cache only ever hits on a *byte-identical* repeat of the
    same file, which almost never happens inside a run. But four of the seven
    sections — the role, the language guidance, the version constraints, and the
    few-shot examples — depend solely on the language pair, the versions, and
    the migration type. They contain no source code at all, yet the old
    single-cache design rebuilt every one of them for every file, because the
    key mixed the code hash into the same lookup.

    Caching those separately turns a batch migration of N files sharing a
    language pair from N full assemblies into one, plus N cheap per-file
    sections. Both caches are keyed by a ``<src>:<srcver>:<tgt>:<tgtver>:<type>``
    path, so :meth:`invalidate` can drop everything derived from one language
    profile through a trie prefix walk when that profile is reloaded.
    """

    def __init__(self, max_cache_entries: int = 100):
        self._cache: PrefixLRU[str] = PrefixLRU(maxsize=max_cache_entries)
        # Far fewer distinct language-pair families than files, so this stays
        # small and hot even when the prompt cache is churning.
        self._sections: PrefixLRU[tuple[str, str]] = PrefixLRU(
            maxsize=max_cache_entries
        )

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
        source_code = source_code[:get_settings().max_llm_code_chars]
        family = self._family_key(
            source_profile,
            target_profile,
            source_version,
            target_version,
            migration_type,
        )
        cache_key = self._cache_key(family, source_code, analyzer_context)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        preamble, few_shots = self._static_sections(
            family,
            source_profile,
            target_profile,
            source_version,
            target_version,
            migration_type,
        )
        prompt = "\n\n".join(
            [
                preamble,
                self._build_analyzer_section(analyzer_context),
                few_shots,
                self._build_source_section(source_profile, source_version, source_code),
                self._build_output_format(),
            ]
        )
        self._cache.set(cache_key, prompt)
        return prompt

    def _static_sections(
        self,
        family: str,
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        source_version: str,
        target_version: str,
        migration_type: str,
    ) -> tuple[str, str]:
        """The code-independent sections, built once per language-pair family.

        Returned as ``(preamble, few_shots)`` because the analyzer section sits
        between them in the final prompt — the ordering is unchanged, only the
        rebuilding is skipped.
        """
        cached = self._sections.get(family)
        if cached is not None:
            return cached

        preamble = "\n\n".join(
            [
                self._build_system_role(source_profile, target_profile, migration_type),
                self._build_language_guidance(
                    source_profile,
                    target_profile,
                    source_version,
                    target_version,
                    migration_type,
                ),
                self._build_version_constraints(
                    target_profile, target_version, migration_type
                ),
            ]
        )
        few_shots = self._build_few_shots(
            source_profile, target_profile, migration_type
        )
        sections = (preamble, few_shots)
        self._sections.set(family, sections)
        return sections

    def cache_size(self) -> int:
        return len(self._cache)

    def section_cache_size(self) -> int:
        return len(self._sections)

    def invalidate(self, source_language: str = "") -> int:
        """Drop cached prompts and sections built from a source language.

        Both caches key on the same path, so a reloaded language profile evicts
        exactly what was built from it instead of clearing everything. An empty
        argument walks from the root and drops all of it. Returns the number of
        full prompts dropped.
        """
        prefix = f"{source_language}:" if source_language else ""
        dropped = self._cache.invalidate_prefix(prefix)
        self._sections.invalidate_prefix(prefix)
        return dropped

    @staticmethod
    def _family_key(
        source_profile: LanguageProfile,
        target_profile: LanguageProfile,
        source_version: str,
        target_version: str,
        migration_type: str,
    ) -> str:
        """Path identifying everything that does not depend on the source code."""
        return ":".join(
            [
                source_profile.language_id,
                source_version or "any",
                target_profile.language_id,
                target_version or "any",
                migration_type,
            ]
        ) + ":"

    @staticmethod
    def _cache_key(
        family: str, source_code: str, analyzer_context: dict[str, Any]
    ) -> str:
        """Full-prompt key: the family path plus a digest of the per-file inputs.

        The code length prefixes the payload so no pair of (code, context) values
        can concatenate into the same string and collide.
        """
        context_json = json.dumps(analyzer_context, sort_keys=True, default=str)
        payload = f"{len(source_code)}|{source_code}|{context_json}"
        return family + hashlib.sha256(payload.encode()).hexdigest()

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

    def _build_version_constraints(
        self,
        target_profile: LanguageProfile,
        target_version: str,
        migration_type: str,
    ) -> str:
        """Hard grounding on the *exact* target version.

        The language-guidance section lists version features to reach for; this
        section adds the negative constraint that actually prevents version
        hallucination — do not emit syntax or APIs newer than ``target_version``,
        even when a newer idiom would be cleaner. RAG evidence and few-shots
        supply what *is* available; this fences off what is not.
        """
        name = target_profile.display_name
        lines = [
            "VERSION CONSTRAINTS — HARD REQUIREMENT",
            f"The output must compile and run on {name} {target_version}.",
            f"- Use ONLY syntax, language features, and standard-library APIs "
            f"available in {name} {target_version} or earlier.",
            f"- Do NOT use anything introduced AFTER {name} {target_version}, "
            "even if it is more modern, shorter, or more idiomatic.",
            "- When unsure whether an API exists in this version, choose the "
            "well-established equivalent that you are certain does.",
        ]
        if migration_type == "upgrade_version":
            lines.append(
                f"- Replace constructs deprecated or removed by {name} "
                f"{target_version} with their supported replacements."
            )
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
            "Return a single JSON object with exactly these two keys and nothing "
            "else:\n"
            '  "plan_summary": one concise sentence describing the changes you made.\n'
            '  "migrated_code": the COMPLETE migrated version of the SOURCE CODE '
            "shown above, as a single string.\n\n"
            "RULES:\n"
            "- Migrate the ACTUAL source code provided above. Do NOT invent new "
            "classes and do NOT output placeholder or example code such as "
            "'class Foo'.\n"
            "- Keep every class, method, field, and behavior from the source; only "
            "modernize syntax and idioms for the target version.\n"
            "- Escape newlines inside string values as \\n and inner double quotes "
            'as \\".\n'
            "- No markdown fences, no prose, no comments outside the JSON.\n\n"
            "Respond using this shape (a schema to follow, NOT a value to copy):\n"
            '{"plan_summary": "<one sentence>", "migrated_code": "<full migrated '
            'source code here>"}'
        )
