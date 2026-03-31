from typing import Dict, Any, List, final

from pydantic import BaseModel


class RAGState(BaseModel):

    query : str

    structured_query : Dict[str, Any] = {}

    query_embeddings : List[float] = []

    retrieved_docs : Dict[str, Any] = {}

    confidence : float = 0.0

    final_answer : str = ""