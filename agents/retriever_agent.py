from dataclasses import dataclass
import logging
from typing import List

from schemas.rag_schema import RetrievedDocument, StructuredQuery
from tools.vector_retriever_tool import VectorRetrieverTool


logger = logging.getLogger(__name__)


@dataclass
class RetrievalDecision:
    """
        Internal decision made by the RetrievalAgent.

        This is intentionally deterministic for now.
        A future implementation can replace the decision logic with
        an LLM-based planner without changing the rest of the pipeline.
    """

    strategy: str   # vector_retrieval, BM25, hybrid
    query: str      # The query to be used for retrieval
    reason: str     # Explanation of why this strategy was chosen


class RetrievalAgent:
    """
        Retrieval orchestration agent.

        Responsibilities:
            - Decide whether retrieval should be performed
            - Select the current retrieval strategy
            - Execute retrieval tools
            - Evaluate retrieval quality
            - Retry using a fallback query when necessary
            - Deduplicate results
            - Normalize final ranks

        Does NOT:
            - Talk directly to Qdrant
            - Generate answers
            - Perform embeddings
            - Implement vector search itself
            - Implement reranking itself

        The agent orchestrates those components through tools.
    """

    VECTOR = "vector"

    def __init__(self, 
                 vector_retriever_tool: VectorRetrieverTool,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 minimum_retrieval_count: int = 1,
                 quality_threshold: float = 0.5,
                 max_attempts: int = 3):

        self.vector_retriever_tool = vector_retriever_tool
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.minimum_retrieval_count = minimum_retrieval_count
        self.quality_threshold = quality_threshold
        self.max_attempts = max_attempts


    # retrieve documents based on the structured query
    def retrieve(self, structured_query: StructuredQuery) -> List[RetrievedDocument]:
        """
            Execute retrieval using the selected strategy.
            Returns a list of RetrievedDocument objects.
        """

        if structured_query is None:
            raise ValueError("structured_query is None")

        logger.info("[RetrievalAgent] Starting retrieval for query='%s'",
                structured_query.original_query)

        decision = self._decide_retrieval_strategy(structured_query)

        logger.info(
            "[RetrievalAgent] strategy=%s query='%s' reason='%s'",
            decision.strategy,
            decision.query,
            decision.reason,
        )

        if decision.strategy != self.VECTOR:
            logger.info("[RetrievalAgent] Retrieval not required for strategy=%s",
                decision.strategy)

            return []

        queries = self._build_query_candidates(structured_query)

        all_documents: List[RetrievedDocument] = []

        for attempt,query in enumerate(queries[: self.max_attempts], start=1):

            logger.info("[RetrievalAgent] Retrieval attempt %d/%d query='%s'", attempt,
                min(len(queries), self.max_attempts),
                query)

            documents = self._retrieve_vector(query)

            all_documents = self._merge_documents(all_documents, documents)

            if self._is_sufficient(all_documents):

                logger.info(
                    "[RetrievalAgent] Sufficient retrieval results after attempt %d", 
                    attempt)
                break

            if attempt < min(len(queries), self.max_attempts):
                logger.info(
                    "[RetrievalAgent] Insufficient retrieval results after attempt %d, " \
                    "retrying with fallback query", attempt)


        final_documents = self._finalize_documents(all_documents)

        logger.info("[RetrievalAgent] Final retrieval count=%d", len(final_documents))

        return final_documents


    # decide retrieval strategy
    def _decide_retrieval_strategy(self, structured_query: StructuredQuery) -> RetrievalDecision:
        """
            Decide whether retrieval is required.

            RoutingAgent already sends only RAG requests here, but this
            additional check makes RetrievalAgent independently safe.

            At this stage vector retrieval is the only available retrieval
            strategy.

            Later this method can select:
                - vector
                - BM25
                - hybrid
                - metadata search
                - SQL
                - etc.
        """

        intent = structured_query.intent.value
        source = structured_query.source.value

        if intent in ("INTERNAL_QUESTION", "DOCUMENT_QUERY"):
            return RetrievalDecision(
                strategy=self.VECTOR,
                query=structured_query.original_query,
                reason=f"Intent '{intent}' requires retrieval",)

        if source == "INTERNAL":
            return RetrievalDecision(
                strategy=self.VECTOR,
                query=structured_query.original_query,
                reason=f"Source '{source}' requires retrieval",)

        # This is defensive. RoutingAgent should normally prevent these queries from reaching RetrievalAgent.
        return RetrievalDecision(
            strategy=self.VECTOR,
            query=structured_query.revised_query,
            reason="Fallback retrieval strategy.",)


    # Query planning / reformulation
    def _build_query_candidates(self, structured_query: StructuredQuery) -> List[str]:
        """
            Build a list of query candidates for retrieval.
            The first query is the original query, followed by fallback queries.
            Returns a list of query strings.
        """

        candidates: List[str] = []

        revised_query = structured_query.revised_query.strip()
        original_query = structured_query.original_query.strip()

        if revised_query:
            candidates.append(revised_query)

        if original_query and original_query != revised_query:
            candidates.append(original_query)

        if not candidates:
            candidates.append(structured_query.topic.strip())

        return candidates   


    # tool execution
    def _retrieve_vector(self, query: str) -> List[RetrievedDocument]:
        """
            Execute vector retrieval using the vector retriever tool.
            Returns a list of RetrievedDocument objects.
        """

        try:
            docs = self.vector_retriever_tool.retrieve(query=query, 
                                                    top_k=self.top_k, 
                                                    score_threshold=self.score_threshold)

            return [
                RetrievedDocument(
                    id=str(doc["id"]),
                    content=doc["content"],
                    metadata=doc.get("metadata", {}),
                    similarity_score=float(doc["similarity_score"]),
                    rank=index)

                    for index, doc in enumerate(docs, start=1)
            ]
        
        except Exception:
            logger.error(f"[RetreiverAgent] Error occurred while retrieving vector documents for query='{query}'")
            raise


    def _merge_documents(self, existing_docs: List[RetrievedDocument], 
                         new_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
            Merge new retrieved documents with existing ones.
            Deduplicate based on document ID and normalize ranks.
            Returns a list of unique RetrievedDocument objects.
        """

        by_id = {doc.id: doc for doc in existing_docs}

        for doc in new_docs:
            existing = by_id.get(doc.id)

            if existing is None:
                by_id[doc.id] = doc

            elif doc.similarity_score > existing.similarity_score:
                    by_id[doc.id] = doc

        merged_docs = list(by_id.values())

        merged_docs.sort(key=lambda d: d.similarity_score, reverse=True)

        return merged_docs


    def _is_sufficient(self, documents: List[RetrievedDocument]) -> bool:
        """
            Check if the retrieved documents meet the sufficiency criteria.
            Returns True if sufficient, False otherwise.
        """

        if len(documents) < self.minimum_retrieval_count:
            return False

        if not documents:
            return False

        best_score = max(
            document.similarity_score
            for document in documents
        )

        return best_score >= self.quality_threshold


    def _finalize_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
            Finalize the retrieved documents by filtering based on score threshold
            and normalizing ranks.
            Returns a list of finalized RetrievedDocument objects.
        """

        final_documents = documents[: self.top_k]

        for rank, document in enumerate(final_documents, start=1,):
            document.rank = rank

        return final_documents