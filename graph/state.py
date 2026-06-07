from pydantic import BaseModel, Field

from schemas.rag_schema import RetrievedDocument, Route, StructuredQuery

class RAGState(BaseModel):

    query: str

    structured_query: StructuredQuery | None = None

    route: Route | None = None

    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)

    confidence: float = 0.0

    final_answer: str | None = None

    error: str | None = None