from .embedding_service import CachedEmbeddings
from .ingestion import IngestionPipeline
from .migration_memory import MigrationMemory

# retrieval_pipeline defines the strategy base + request type the concrete
# strategies import, so it must load before them.
from .retrieval_pipeline import RAGPipeline, RetrievalRequest, RetrievalStrategy
from .contextual_compression import ContextualCompressionStrategy
from .corrective_rag import CorrectiveRAGStrategy
from .hyde import HyDEStrategy
from .multi_hop import MultiHopStrategy
from .multi_query import MultiQueryStrategy
from .parent_retriever import ParentDocumentStrategy
from .self_rag import SelfRAGStrategy
from .vector_store import VectorStore

__all__ = [
    "RAGPipeline",
    "RetrievalRequest",
    "RetrievalStrategy",
    "VectorStore",
    "CachedEmbeddings",
    "IngestionPipeline",
    "MigrationMemory",
    # Agentic retrieval strategies (Dimension 2)
    "HyDEStrategy",
    "MultiQueryStrategy",
    "MultiHopStrategy",
    "ContextualCompressionStrategy",
    "ParentDocumentStrategy",
    "CorrectiveRAGStrategy",
    "SelfRAGStrategy",
]
