import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Language-aware separators — more aggressive for structured languages
LANGUAGE_SEPARATORS: dict[str, list[str]] = {
    "python": ["\n\nclass ", "\n\ndef ", "\n    def ", "\n\n", "\n", " ", ""],
    "java": ["\n\npublic ", "\n\nprivate ", "\n\nprotected ", "\n\nclass ", "\n\ninterface ", "\n\n", "\n", " ", ""],
    "javascript": ["\n\nfunction ", "\n\nclass ", "\n\nconst ", "\n\nlet ", "\n\n", "\n", " ", ""],
    "typescript": ["\n\nfunction ", "\n\nclass ", "\n\ninterface ", "\n\nconst ", "\n\n", "\n", " ", ""],
    "csharp": ["\n\npublic ", "\n\nprivate ", "\n\nclass ", "\n\nstruct ", "\n\n", "\n", " ", ""],
    "go": ["\n\nfunc ", "\n\ntype ", "\n\n", "\n", " ", ""],
    "kotlin": ["\n\nfun ", "\n\nclass ", "\n\n", "\n", " ", ""],
    "rust": ["\n\nfn ", "\n\nstruct ", "\n\nimpl ", "\n\n", "\n", " ", ""],
    "cpp": ["\n\nclass ", "\n\nstruct ", "\n\nvoid ", "\n\nint ", "\n\n", "\n", " ", ""],
}

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


class DocSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document], language: str) -> list[Document]:
        separators = LANGUAGE_SEPARATORS.get(language, DEFAULT_SEPARATORS)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len,
        )
        # Split per source document so each chunk can be linked back to its parent
        # (parent_id) and ordered within it (chunk_index). The Parent Document
        # retrieval strategy uses these to reassemble the full parent from a child
        # hit; both are Chroma-safe scalars. Iterating documents in order preserves
        # the overall chunk sequence the previous batch call produced.
        chunks: list[Document] = []
        for document in documents:
            parent_id = self._parent_id(document)
            doc_chunks = splitter.split_documents([document])
            for index, chunk in enumerate(doc_chunks):
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                chunk.metadata["parent_id"] = parent_id
                chunk.metadata["chunk_index"] = index
                if index > 0:
                    # Only chunks after the first need a header: the first
                    # already begins with the document's own title.
                    chunk.page_content = self._with_context(chunk, language)
            chunks.extend(doc_chunks)
        return chunks

    @staticmethod
    def _with_context(chunk: Document, language: str) -> str:
        """Prefix a chunk with what it is, so it can stand alone.

        A chunk taken from the middle of a document arrives with no idea what it
        is about: a fragment reading "use ``Executors.newVirtualThreadPerTask``"
        never says which language or version it belongs to. That hurts twice —
        the embedding is computed from text missing its own subject, and the
        model reads the fragment out of context.

        The header is cheap (a line or two) and made of metadata the ingestion
        pipeline already attached, so it costs no extra call.
        """
        md = chunk.metadata or {}
        bits = [str(md.get("language", language) or language)]
        version = md.get("version")
        if version and version != "any":
            bits.append(str(version))
        doc_type = md.get("doc_type")
        if doc_type:
            bits.append(str(doc_type).replace("-", " "))

        header = f"[{' · '.join(bits)}]"
        title = md.get("title") or md.get("source", "")
        if title:
            # Just the filename: a full path is noise in an embedding.
            header += f" {str(title).replace(chr(92), '/').rsplit('/', 1)[-1]}"
        return f"{header}\n{chunk.page_content}"

    @staticmethod
    def _parent_id(document: Document) -> str:
        """Stable id for the source document a chunk belongs to.

        Derived from the source path (when known) plus a hash of the full content,
        so two distinct source docs never collide and re-ingesting the same doc
        yields the same parent id.
        """
        source = (document.metadata or {}).get("source", "")
        digest = hashlib.sha256(
            f"{source}\n{document.page_content}".encode()
        ).hexdigest()
        return digest[:16]
