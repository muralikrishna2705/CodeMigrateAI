import logging

from langchain_core.documents import Document

from .loaders import DocLoader
from .splitter import DocSplitter

log = logging.getLogger("CodeMigrateAI.DocumentPipeline")


class DocumentPipeline:
    def __init__(self, loader: DocLoader, splitter: DocSplitter):
        self.loader = loader
        self.splitter = splitter

    async def process_language(self, language: str, source_types: list[str]) -> list[Document]:
        all_chunks: list[Document] = []
        for source_type in source_types:
            docs = await self.loader.load_source(source_type, language)
            if docs:
                chunks = self.splitter.split(docs, language)
                all_chunks.extend(chunks)
                log.info("  %s: %d docs -> %d chunks", source_type, len(docs), len(chunks))
        return all_chunks
