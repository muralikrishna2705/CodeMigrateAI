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
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        return chunks
