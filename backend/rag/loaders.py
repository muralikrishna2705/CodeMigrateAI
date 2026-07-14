import asyncio
import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

log = logging.getLogger("CodeMigrateAI.Loaders")

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

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
        for doc in docs:
            doc.metadata["source_type"] = subdir
            doc.metadata["language"] = language
        log.info("Loaded %d %s docs for %s", len(docs), subdir, language)
        return docs
