import logging

from rag.retriever import Retriever
from tools.retriever_tool import RetrieverTool


logger = logging.getLogger(__name__)


class VectorRetrieverTool(RetrieverTool):
    '''
        Thin wrapper around the vector store.

        Responsibilities:
            - Execute similarity search
            - Return raw retrieval results
    '''
    def __init__(self, retriever: Retriever):
        self.retriever = retriever


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> list[dict]:
        '''
            - Take in a query
            - Generate embedding
            - Execute similarity search
            - Return raw retrieval results
        '''
        logger.info(f"[VectorRetrievalTool] Retrieving documents for query='{query}'")

        try:
            return self.retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        
        except Exception as e:
            logger.error(f"[VectorRetrievalTool] Error occurred while retrieving documents: {e}")
            raise