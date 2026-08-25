from rag.BM25store import BM25Store
from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingPipeline

# single instances
vector_store = VectorStore()
embedding_pipeline = EmbeddingPipeline()
bm25_store = BM25Store()
