from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LanguageProfile:
    language_id: str
    display_name: str
    versions: list[str]

    syntax_rules: list[str] = field(default_factory=list)
    idioms: dict[str, str] = field(default_factory=dict)
    stdlib_mappings: dict[str, str] = field(default_factory=dict)
    version_features: dict[str, list[str]] = field(default_factory=dict)
    few_shot_examples: list[dict] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)
    style_guide: str = ""
    cross_language_mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    def get_version_features(self, version: str) -> list[str]:
        return self.version_features.get(version, [])

    def get_cross_language_mapping(self, target_language: str) -> dict[str, str]:
        return self.cross_language_mappings.get(target_language.lower(), {})

    def get_system_role(self, migration_type: str) -> str:
        if migration_type == "upgrade_version":
            return f"You are an expert {self.display_name} modernization engineer."
        return f"You are an expert polyglot engineer fluent in {self.display_name}."


class ProfileRegistry:
    _profiles: dict[str, LanguageProfile] = {}
    _aliases: dict[str, str] = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "ts": "typescript",
        "c#": "csharp",
        "cs": "csharp",
        "c++": "cpp",
        "golang": "go",
    }

    @classmethod
    def normalize(cls, language_id: str) -> str:
        key = language_id.strip().lower()
        return cls._aliases.get(key, key)

    @classmethod
    def register(cls, profile: LanguageProfile):
        cls._profiles[profile.language_id] = profile

    @classmethod
    def get(cls, language_id: str) -> Optional[LanguageProfile]:
        return cls._profiles.get(cls.normalize(language_id))

    @classmethod
    def all(cls) -> dict[str, LanguageProfile]:
        return cls._profiles.copy()
