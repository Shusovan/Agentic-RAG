from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, List

from schemas.rag_schema import RetrievedDocument, StructuredQuery
from tools.BM25_retriever_tool import BM25RetrieverTool
from tools.hybrid_retriever_tool import HybridRetrieverTool
from tools.vector_retriever_tool import VectorRetrieverTool


logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy(str, Enum):

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RetrievalAgent:
    """
        Agentic retrieval orchestration layer.

        Responsibilities:

            1. Decide whether retrieval is required
            2. Select retrieval strategy
            3. Select retrieval tool
            4. Execute retrieval
            5. Evaluate retrieval quality
            6. Reformulate query when necessary
            7. Retry using another strategy
            8. Optionally rerank
            9. Return normalized RetrievedDocument objects

        Retrieval tools themselves do NOT make strategy decisions.
    """

    def __init__(self, 
                 vector_retriever_tool: VectorRetrieverTool,
                 bm25_retriever_tool: BM25RetrieverTool,
                 hybrid_retriever_tool: HybridRetrieverTool,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 minimum_retrieval_count: int = 1,
                 quality_threshold: float = 0.5,
                 max_attempts: int = 3):

        self.vector_retriever_tool = vector_retriever_tool
        self.bm25_retriever_tool = bm25_retriever_tool
        self.hybrid_retriever_tool = hybrid_retriever_tool
        self.top_k = top_k
        self.minimum_retrieval_count = minimum_retrieval_count
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts


    def retrieve(
        self,
        structured_query: StructuredQuery
    ) -> List[RetrievedDocument]:

        logger.info(
            "[RetrievalAgent] Starting retrieval for '%s'",
            structured_query.revised_query,
        )

        if not self._retrieval_decision(structured_query):
            logger.info("[RetrievalAgent] Retrieval not required")
            return []

        strategies = self._plan_retrieval(structured_query)

        current_query = structured_query.revised_query

        for attempt, strategy in enumerate(
            strategies[:self.max_attempts],
            start=1
        ):

            logger.info(
                "[RetrievalAgent] Attempt %d/%d strategy=%s query='%s'",
                attempt,
                self.max_attempts,
                strategy,
                current_query,
            )

            results = self._execute_strategy(
                strategy=strategy,
                query=current_query
            )

            logger.info(
                "[RetrievalAgent] Strategy=%s returned %d results",
                strategy,
                len(results),
            )

            if self._is_sufficient(results):

                results = self._rerank(
                    query=current_query,
                    results=results
                )

                return self._normalize_results(results)

            logger.warning(
                "[RetrievalAgent] Strategy=%s produced insufficient results",
                strategy,
            )

            # Try next strategy with a reformulated query
            current_query = self._reformulate_query(
                structured_query,
                previous_query=current_query,
            )

        logger.warning(
            "[RetrievalAgent] Retrieval exhausted all strategies"
        )

        return []


    # retrieve documents based on the structured query
    # def retrieve(self, structured_query: StructuredQuery) -> List[RetrievedDocument]:
    #     """
    #         Execute retrieval using the selected strategy.
    #         Returns a list of RetrievedDocument objects.
    #     """

    #     logger.info("[RetrievalAgent] Starting retrieval for '%s'",
    #         structured_query.revised_query,)

    #     if not self._retrieval_decision(structured_query):
    #         logger.info("[RetrievalAgent] Retrieval not required")
    #         return []

    #     # plan retrieval strategy
    #     strategies = self._plan_retrieval(structured_query)

    #     current_query = structured_query.revised_query

    #     for attempt, strategy in enumerate(strategies[:self.max_attempts], start=1):

    #         logger.info("[RetrievalAgent] Attempt %d/%d strategy=%s",
    #             attempt,
    #             self.max_attempts,
    #             strategy, current_query) 

    #         results = self._execute_strategy(strategy=strategy, query=current_query)

    #         logger.info("[RetrievalAgent] Strategy=%s returned %d results",
    #         strategy,
    #         len(results),)

    #         if self._is_sufficient(results):
    #             results = self._rerank(query=current_query, results=results)

    #         return self._normalize_results(results)

    #     logger.warning("[RetrievalAgent] Insufficient results for strategy=%s", strategy,)

    #     # Reformulate before next attempt
    #     current_query = self._reformulate_query(structured_query, previous_query=current_query,)

    #     logger.warning("[RetrievalAgent] Retrieval exhausted all strategies")

    #     return []


    # retrieval decision
    def _retrieval_decision(self, structured_query: StructuredQuery) -> bool:

        intent = structured_query.intent.value

        return intent in {"INTERNAL_QUESTION", "DOCUMENT_QUERY"}


    # retrieval planning
    def _plan_retrieval(self, structured_query: StructuredQuery) -> List[RetrievalStrategy]:

        query = structured_query.revised_query.lower()

        # Exact identifiers / codes / filenames
        lexical_indicators = [
            "filename",
            "file",
            "document id",
            "policy id",
            "employee id",
            "invoice",
            "contract",
            "section",
            "clause",
            "version",
            "error code",
        ]

        if any (indicator in query for indicator in lexical_indicators):
            return [
                RetrievalStrategy.BM25,
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.VECTOR
            ]

        return [
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.VECTOR,
            RetrievalStrategy.BM25
        ]


    # execute selective retrieval
    def _execute_strategy(self, strategy: RetrievalStrategy, query: str) -> List[Dict[str, Any]]:

        if strategy == RetrievalStrategy.VECTOR:
            return self.vector_retriever_tool.retrieve(query=query, top_k=self.top_k)

        if strategy == RetrievalStrategy.BM25:
            return self.bm25_retriever_tool.retrieve(query=query, top_k=self.top_k)

        if strategy == RetrievalStrategy.HYBRID:
            return self.hybrid_retriever_tool.retrieve(query=query, top_k=self.top_k)

        raise ValueError(f"Unsupported retrieval strategy: {strategy}")


    # evaluate retrieved quality
    def _is_sufficient(self, results: List[Dict[str, Any]]) -> bool:

        if len(results) < self.minimum_retrieval_count:
            return False

        if not results:
            return False

        top_score = results[0].get("score", 0.0)

        if top_score <= 0:
            return False

        return True


    # query reformulation
    def _reformulate_query(self, structured_query: StructuredQuery, previous_query: str) -> str:

        entities = "".join(structured_query.entities)

        topic = structured_query.topic

        reformulated = (f"{topic} {entities} {structured_query.original_query}")

        # Remove duplicated whitespace
        return " ".join(reformulated.split())


    # re-ranking
    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        if len(results) < 3:
            return results

        # write the rerank code here
        # cross encoders

        return results


    def _normalize_results(self, results: List[Dict[str, Any]]) -> List[RetrievedDocument]:

        documents = []

        for rank, result in enumerate(results, start=1):

            # For hybrid retrieval, use the original vector similarity.
            # Fall back to score for BM25/vector-only retrieval.
            score = float(
                result.get(
                    "vector_score",
                    result.get("similarity_score",
                            result.get("score", 0.0))
                )
            )

            # Keep similarity score in [0, 1]
            score = max(0.0, min(1.0, score))

            documents.append(
                RetrievedDocument(
                    id=str(result["id"]),
                    content=result["content"],
                    metadata=result.get("metadata", {}),
                    similarity_score=score,
                    rank=rank,
                )
            )

        logger.info(
            "[RetrievalAgent] Normalized %d documents",
            len(documents)
        )

        return documents


    # normalize results
    # def _normalize_results(self, results: List[Dict[str, Any]]) -> List[RetrievedDocument]:

    #     documents = []

    #     for rank, result in enumerate(results, start=1):

    #         score = float(result.get("score", 0.0))

    #         score = max(0.0, min(1.0, score))

    #         documents.append(
    #             RetrievedDocument(
    #                 id=str(result["id"]),
    #                 content=result["content"],
    #                 metadata=result.get(
    #                     "metadata",
    #                     {},
    #                 ),
    #                 similarity_score=score,
    #                 rank=rank,
    #             )
    #         )

    #     logger.info("Normalized Documents", documents)

    #     return documents