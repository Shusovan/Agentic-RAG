from graph.workflow import Flow
from schemas.rag_schema import Intent, RAGResponse, Route, Source, StructuredQuery


class OrchestratorAgent:

    def __init__(self):
        self.flow = Flow()

    def run(self, query: str):
        return self.flow.run(query)

    # def run(self, query) -> RAGResponse:

    #     structured: StructuredQuery = self.query_understanding_agent.process_query(query)

    #     route = self._routing_agent(structured)

    #     result = self._execute_route(route, structured)

    #     return result
    

    # def _routing_agent(self, structured: StructuredQuery) -> Route:

    #     if structured.intent == Intent.OUT_OF_SCOPE:
    #         return Route.FALLBACK

    #     if structured.intent == Intent.AMBIGUOUS:
    #         return Route.CLARIFY

    #     if structured.intent in (Intent.INTERNAL_QUESTION, Intent.DOCUMENT_QUERY):
    #         return Route.RAG

    #     if structured.source == Source.INTERNAL:
    #         return Route.RAG

    #     if structured.intent in (Intent.GENERAL_QUESTION, Intent.GREETING, Intent.CHITCHAT):
    #         return Route.LLM

    #     if structured.source == Source.GENERAL:
    #         return Route.LLM

    #     return Route.CLARIFY
    

    # def _execute_route(
    #     self,
    #     route: Route,
    #     structured: StructuredQuery,
    # ) -> RAGResponse:

    #     if route == Route.RAG:

    #         documents = self.vector_retriever_agent.retrieve(structured)

    #         answer = self.answer_generation_agent.generate_rag_answer(
    #             structured_query=structured,
    #             documents=documents,
    #         )

    #         return RAGResponse(
    #             route=route,
    #             answer=answer,
    #             documents=documents,
    #         )

    #     if route == Route.LLM:

    #         answer = self.answer_generation_agent.generate_general_answer(
    #             query=structured.original_query
    #         )

    #         return RAGResponse(
    #             route=route,
    #             answer=answer,
    #             documents=[],
    #         )

    #     if route == Route.CLARIFY:

    #         return RAGResponse(
    #             route=route,
    #             answer="Could you clarify your question?",
    #             documents=[],
    #         )

    #     return RAGResponse(
    #         route=Route.FALLBACK,
    #         answer="This request is outside the supported scope.",
    #         documents=[],
    #     )