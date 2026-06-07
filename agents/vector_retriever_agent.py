import logging

from schemas.rag_schema import RetrievedDocument, StructuredQuery
from tools.vector_retriever_tool import VectorRetrieverTool


logger = logging.getLogger(__name__)


class VectorRetrieverAgent:
    """
    Retrieval orchestration layer.

    Responsibilities:
        - Decide retrieval query
        - Call VectorRetrievalTool
        - Convert results into RetrievedDocument
        - Apply any future retrieval policies

    Does NOT:
        - Talk directly to Qdrant
        - Generate answers
        - Route requests
    """
    def __init__(self, retriever_tool: VectorRetrieverTool, top_k: int = 5, score_threshold: float = 0.0):
        self.retriever_tool = retriever_tool
        self.top_k = top_k
        self.score_threshold = score_threshold
        

    def retrieve(self, structured_query: StructuredQuery) -> list[RetrievedDocument]:
        logger.info(f"[VectorRetrieverAgent] " f"Retrieving documents for query='{structured_query.revised_query}'")

        try:
            results = self.retriever_tool.retrieve(
                query=structured_query.revised_query,
                top_k=self.top_k,
                score_threshold=self.score_threshold)
            
            retrieved_docs = [
                RetrievedDocument(
                    id=result["id"],
                    content=result["content"],
                    metadata=result["metadata"],
                    similarity_score=result["similarity_score"],
                    rank=result["rank"],
                )
                for result in results
            ]

            logger.info(f"[VectorRetrieverAgent] " f"Retrieved {len(retrieved_docs)} documents")

            return retrieved_docs

        except Exception:
            logger.exception("[VectorRetrieverAgent] Retrieval failed")
            raise