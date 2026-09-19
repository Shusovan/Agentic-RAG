import logging

from llm.generative_llm import GenerativeLLM
from schemas.rag_schema import RetrievedDocument, StructuredQuery


logger = logging.getLogger(__name__)


class AnswerGenerationAgent:
    """
        Generates the final answer.

        Responsibilities:
            - Format retrieved context
            - Call GenerativeLLM
            - Return generated answer

        Does NOT:
            - Retrieve documents
            - Route requests
            - Talk directly to vector stores
    """

    def __init__(self, llm: GenerativeLLM):
        self.llm = llm


    def generate_rag_answer(self, structured_query: StructuredQuery, documents: list[RetrievedDocument],) -> str:

        logger.info("[AnswerGenerationAgent] Generating RAG answer")

        context = self._build_context(documents)

        return self.llm.generate_response(query=structured_query.original_query,context=context,)


    def generate_general_answer(self, query: str,) -> str:

        logger.info("[AnswerGenerationAgent] Generating general answer")

        return self.llm.generate_response(query=query)


    def _build_context(self, documents: list[RetrievedDocument],) -> str:

        if not documents:
            return ""

        context_chunks = []

        for doc in documents:
            context_chunks.append(f"[Document {doc.rank}]\n{doc.content}")

        return "\n\n".join(context_chunks)