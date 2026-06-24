from .base import LanguageProfile, ProfileRegistry
from .cpp import CppProfile
from .csharp import CSharpProfile
from .go import GoProfile
from .java import JavaProfile
from .javascript import JavaScriptProfile
from .kotlin import KotlinProfile
from .python import PythonProfile
from .rust import RustProfile
from .typescript import TypeScriptProfile


def _register_defaults():
    for profile in (
        JavaProfile(),
        PythonProfile(),
        JavaScriptProfile(),
        TypeScriptProfile(),
        CSharpProfile(),
        GoProfile(),
        KotlinProfile(),
        RustProfile(),
        CppProfile(),
    ):
        ProfileRegistry.register(profile)


_register_defaults()

LANGUAGE_PROFILES = ProfileRegistry.all()


def get_profile(language_id: str) -> LanguageProfile:
    profile = ProfileRegistry.get(language_id)
    if profile is None:
        supported = ", ".join(sorted(ProfileRegistry.all()))
        raise ValueError(
            f"Unsupported language '{language_id}'. Supported languages: {supported}"
        )
    return profile


def get_supported_profiles() -> dict[str, LanguageProfile]:
    return ProfileRegistry.all()


__all__ = [
    "LANGUAGE_PROFILES",
    "LanguageProfile",
    "ProfileRegistry",
    "get_profile",
    "get_supported_profiles",
]
