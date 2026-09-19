from schemas.rag_schema import (
    Intent,
    Route,
    Source,
    StructuredQuery,
)


class RoutingAgent:

    def route(self, structured: StructuredQuery,) -> Route:

        if structured.intent == Intent.OUT_OF_SCOPE:
            return Route.FALLBACK

        if structured.intent == Intent.AMBIGUOUS:
            return Route.CLARIFY

        if structured.intent in (
            Intent.INTERNAL_QUESTION,
            Intent.DOCUMENT_QUERY,
        ):
            return Route.RAG

        if structured.source == Source.INTERNAL:
            return Route.RAG

        if structured.intent in (
            Intent.GENERAL_QUESTION,
            Intent.GREETING,
            Intent.CHITCHAT,
        ):
            return Route.LLM

        if structured.source == Source.GENERAL:
            return Route.LLM

        return Route.CLARIFY