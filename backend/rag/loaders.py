import asyncio
import logging
import re
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from config import get_settings

log = logging.getLogger("CodeMigrateAI.Loaders")

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# --- Document classification -----------------------------------------------
#
# Version-aware retrieval needs every chunk tagged with the version, doc type,
# and authority of the doc it came from. We derive those from the corpus layout
# so no separate manifest has to be maintained:
#
#   reference/<lang>/<bucket>/[<version>/][<doc_type>/]file.ext
#
# Both the <version> and <doc_type> path segments are OPTIONAL — existing flat
# content (reference/python/examples/foo.py) keeps working and is tagged with
# the wildcard version so it still matches the version-filtered retrieval leg.

# A path segment is a version if it looks like one: "21", "3.12", "1.80",
# "5.x", "ES2020". Anything else is treated as an organisational sub-folder.
_VERSION_RE = re.compile(r"^v?\d+(?:\.\d+){0,2}(?:\.x)?$|^ES\w+$", re.IGNORECASE)

# Recognised doc-type folder names (normalised) -> canonical doc_type.
_DOC_TYPE_ALIASES = {
    "example": "example",
    "examples": "example",
    "user": "user",
    "reference": "reference",
    "ref": "reference",
    "stdlib": "reference",
    "api": "api",
    "tutorial": "tutorial",
    "guide": "tutorial",
    "migration": "migration-guide",
    "migration-guide": "migration-guide",
    "migrationguide": "migration-guide",
    "release-notes": "release-notes",
    "releasenotes": "release-notes",
    "changelog": "release-notes",
    "deprecation": "deprecation",
    "deprecations": "deprecation",
    "deprecated": "deprecation",
}

# Per-bucket defaults: (doc_type, is_official). Fetched docs come from the
# official documentation URLs, so they carry the highest authority.
_BUCKET_DEFAULTS: dict[str, tuple[str, bool]] = {
    "examples": ("example", False),
    "user": ("user", False),
    "_fetched": ("reference", True),
}


def derive_doc_metadata(
    relative_parts: list[str], bucket: str, wildcard: str = "any"
) -> dict:
    """Classify a doc from its path segments below the bucket directory.

    ``relative_parts`` are the folder names between ``reference/<lang>/<bucket>``
    and the file itself (filename excluded). A version-looking segment sets the
    version; a recognised doc-type segment overrides the bucket default. Missing
    segments leave the wildcard version / bucket defaults in place.
    """
    doc_type, is_official = _BUCKET_DEFAULTS.get(bucket, ("reference", False))
    version = wildcard
    for part in relative_parts:
        token = part.strip()
        if not token:
            continue
        if _VERSION_RE.match(token):
            version = token
            continue
        alias = _DOC_TYPE_ALIASES.get(token.lower())
        if alias:
            doc_type = alias
    return {"version": version, "doc_type": doc_type, "is_official": is_official}

# One glob per extension. A single brace pattern ("**/*.{py,java,...}") does NOT
# work here: DirectoryLoader delegates to pathlib.Path.glob, which does not
# expand shell-style braces, so the brace form silently matches nothing.
_SOURCE_GLOBS = [
    "**/*.py",
    "**/*.java",
    "**/*.js",
    "**/*.ts",
    "**/*.cs",
    "**/*.go",
    "**/*.kt",
    "**/*.rs",
    "**/*.cpp",
    "**/*.md",
    "**/*.txt",
]


class DocLoader:
    def __init__(self):
        self._built_in_dir = REFERENCE_DIR

    async def load_source(self, source_type: str, language: str) -> list[Document]:
        """Load documents from one of 3 sources."""
        if source_type == "builtin":
            return await self._load_directory(language, "examples")
        elif source_type == "user":
            return await self._load_directory(language, "user")
        elif source_type == "fetched":
            return await self._load_directory(language, "_fetched")
        return []

    async def _load_directory(self, language: str, subdir: str) -> list[Document]:
        target = self._built_in_dir / language / subdir
        if not target.exists():
            log.debug("No %s docs for %s at %s", subdir, language, target)
            return []

        # LangChain's DirectoryLoader is synchronous; wrap in executor
        loop = asyncio.get_event_loop()
        loader = DirectoryLoader(
            str(target),
            glob=_SOURCE_GLOBS,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = await loop.run_in_executor(None, loader.load)
        wildcard = get_settings().rag_version_wildcard
        for doc in docs:
            rel_parts = self._relative_parts(target, doc.metadata.get("source"))
            meta = derive_doc_metadata(rel_parts, subdir, wildcard)
            doc.metadata["source_type"] = subdir
            doc.metadata["language"] = language
            doc.metadata["version"] = meta["version"]
            doc.metadata["doc_type"] = meta["doc_type"]
            doc.metadata["is_official"] = meta["is_official"]
        log.info("Loaded %d %s docs for %s", len(docs), subdir, language)
        return docs

    @staticmethod
    def _relative_parts(base_dir: Path, source_path: str | None) -> list[str]:
        """Folder segments between ``base_dir`` and the file (filename dropped)."""
        if not source_path:
            return []
        try:
            rel = Path(source_path).resolve().relative_to(base_dir.resolve())
        except (ValueError, OSError):
            return []
        return list(rel.parts[:-1])
