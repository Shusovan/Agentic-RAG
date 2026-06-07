import logging
from typing import List, Dict, Any


from rag.embeddings import EmbeddingPipeline
from rag.vectorstore import VectorStore


logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles query-based retrieval from the vector store
    """

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingPipeline):
        """
        Initialize Retriever

        Args:
            vector_store : VectorStore instance (Qdrant backend)
            embedding_manager : EmbeddingPipeline instance
        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query
        """

        try:

            logger.info("Generating query embedding")

            embedding = self.embedding_manager.embed_documents([query])[0]

            return embedding.tolist()

        except Exception as e:

            raise ValueError(
                f"Query embedding failed: {e}"
            )


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents

        Args:
            query : user query
            top_k : number of results
            score_threshold : minimum similarity score

        Returns:
            List of retrieved documents
        """

        logger.info(f"Retrieving documents for query='{query}', top_k={top_k}")

        try:

            query_embedding = self._embed_query(query)

            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k
            )

            retrieved_docs = []

            for rank, result in enumerate(results, start=1):

                score = result["score"]

                if score >= score_threshold:

                    retrieved_docs.append({
                        "id": result["id"],
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "similarity_score": score,
                        "rank": rank
                    })

            logger.info(
                f"Retrieved {len(retrieved_docs)} documents"
            )

            return retrieved_docs

        except Exception as e:

            raise ValueError(
                f"Document retrieval failed: {e}"
            )
