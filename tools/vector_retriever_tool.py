import logging

from rag.retriever import Retriever
from tools.retriever_tool import RetrieverTool


logger = logging.getLogger(__name__)


class VectorRetrieverTool(RetrieverTool):
    """
        Tool responsible for executing vector similarity retrieval.

        The tool does not decide:
            - whether retrieval is needed
            - which retrieval strategy to use
            - whether results are sufficient
            - whether to retry

        Those decisions belong to RetrievalAgent.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:

        logger.info(f"[VectorRetrievalTool] Retrieving documents for query='{query}'")

        try:
            return self.retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        
        except Exception as e:
            logger.error(f"[VectorRetrievalTool] Error occurred while retrieving documents: {e}")
            raise