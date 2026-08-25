import logging
from typing import Any, Dict, List

from rag.BM25store import BM25Store
from tools.retriever_tool import RetrieverTool


logger = logging.getLogger(__name__)


class BM25RetrieverTool(RetrieverTool):
    """
        Lexical retrieval tool.

        Searches the BM25 index over the same chunks
        stored in the vector database.
    """

    name = "BM25RetrieverTool"
    description = "A tool for retrieving relevant documents using BM25 algorithm."

    def __init__(self, bm25_store: BM25Store):

        self.bm25_store = bm25_store


    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        logger.info("[BM25RetrieverTool] query='%s', top_k=%d", query, top_k,)
        
        return self.bm25_store.query(query=query, top_k=top_k)
    
    