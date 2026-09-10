import logging
from typing import Any, Dict, List, Set

from rag.retriever import Retriever
from rag.retrieval_metrics import RetrievalMetrics


logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
        Evaluates a Retriever against a ground-truth evaluation dataset.

        Evaluation is performed at two levels:

        1. Document level
        - document_id
        - Measures whether the correct source document was retrieved.

        2. Chunk level
        - chunk_id
        - Measures whether the specific relevant chunk was retrieved.

        The production Retriever itself is not modified.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever


    # SINGLE QUERY
    def evaluate_query(self, query: str, ground_truth: Dict[str, Any],
        k_values: List[int],) -> Dict[str, Any]:
        """
            Evaluate a single query.

            Expected ground_truth format:

            {
                "document_id": "abc123",
                "chunk_ids": [
                    "abc123_chunk_7"
                ]
            }

            Multiple relevant documents/chunks are also supported:

            {
                "document_ids": ["abc123", "xyz456"],
                "chunk_ids": [
                    "abc123_chunk_7",
                    "xyz456_chunk_3"
                ]
            }
        """

        if not query or not query.strip():
            raise ValueError("Evaluation query cannot be empty")

        if not ground_truth:
            raise ValueError("Ground truth cannot be empty")

        max_k = max(k_values)

        logger.info("Evaluating Query | top_k=%s | query=%s", max_k, query,)

        # Retrieve
        results = self.retriever.retrieve(query=query, top_k=max_k, score_threshold=0.0,)

        # Extract retrieved document/chunk IDs
        retrieved_document_ids = list(dict.fromkeys(str(result["document_id"])
            for result in results
            if result.get("document_id")))

        # Extract retrieved chunks
        retrieved_chunk_ids = [str(result["chunk_id"])
            for result in results
            if result.get("chunk_id")]

        # Ground truth document IDs
        if "document_ids" in ground_truth:
            relevant_document_ids: Set[str] = {str(document_id)
                for document_id in ground_truth["document_ids"]}

        elif "document_id" in ground_truth:
            relevant_document_ids = {str(ground_truth["document_id"])}

        else:
            relevant_document_ids = set()

        # Ground truth chunk IDs
        relevant_chunk_ids: Set[str] = {str(chunk_id)
            for chunk_id in ground_truth.get("chunk_ids", [])}

        # Calculate metrics
        metrics = {}

        for k in k_values:

            # Document-level metrics (Precision@k)
            metrics[f"Document_Precision@{k}"] = (RetrievalMetrics.precision_at_k(
                    retrieved_ids=retrieved_document_ids,
                    relevant_ids=relevant_document_ids,
                    k=k,))

            metrics[f"Document_Recall@{k}"] = (RetrievalMetrics.recall_at_k(
                    retrieved_ids=retrieved_document_ids,
                    relevant_ids=relevant_document_ids,
                    k=k,))

            metrics[f"Document_Hit@{k}"] = (RetrievalMetrics.hit_at_k(
                retrieved_ids=retrieved_document_ids,
                relevant_ids=relevant_document_ids,
                k=k
            ))


            # Chunk-level metrics
            if relevant_chunk_ids:

                metrics[f"Chunk_Precision@{k}"] = (RetrievalMetrics.precision_at_k(
                        retrieved_ids=retrieved_chunk_ids,
                        relevant_ids=relevant_chunk_ids,
                        k=k,))

                metrics[f"Chunk_Recall@{k}"] = (RetrievalMetrics.recall_at_k(
                        retrieved_ids=retrieved_chunk_ids,
                        relevant_ids=relevant_chunk_ids,
                        k=k,))

                metrics[f"Chunk_Hit@{k}"] = (RetrievalMetrics.hit_at_k(
                    retrieved_ids=retrieved_chunk_ids,
                    relevant_ids=relevant_chunk_ids,
                    k=k,))

        retrieved_data = []

        for rank, result in enumerate(results, start=1):
            retrieved_data.append(
                {
                    "rank": rank,
                    "document_id": (
                        str(result["document_id"] if result.get("document_id") else None)
                    ),
                    "chunk_id": (
                        str(result["chunk_id"] if result.get("chunk_id") else None)
                    ),
                    "similarity_score": result.get("similarity_score"),
                    "content": result.get("content", ""),
                }
            )


        # Return detailed query result
        return {
            "query": query,
            "ground_truth": 
            {
                "document_ids": list(relevant_document_ids),
                "chunk_ids": list(relevant_chunk_ids),
            },
            "retrieved": 
            {
                "document_ids": retrieved_document_ids,
                "chunk_ids": retrieved_chunk_ids,
                "retrieved_data": retrieved_data
            },
            "metrics": metrics,
        }


    # DATASET
    def eval_dataset(self, dataset: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, Any]:
        """
        Evaluate the complete evaluation dataset.

        Expected dataset format:

        [
            {
                "query_id": "q_001",
                "query": "...",
                "ground_truth": {
                    "document_id": "abc123",
                    "chunk_ids": [
                        "abc123_chunk_7"
                    ]
                }
            }
        ]
        """

        query_results = []

        for index, item in enumerate(dataset, start=1):

            query = item.get("query")

            logger.info("Evaluating query %s/%s", index, len(dataset),)

            try:
                ground_truth = item.get("ground_truth", {})

                result = self.evaluate_query(query=query, ground_truth=ground_truth,
                    k_values=k_values,)

                result["query_id"] = item.get("query_id",f"q_{index:03d}",)

                result["status"] = "success"

                query_results.append(result)

            except Exception as exc:
                logger.exception("Evaluation failed for query: %s", query,)
                query_results.append(
                    {
                        "query_id": item.get(
                            "query_id",
                            f"q_{index:03d}",
                        ),
                        "query": query,
                        "status": "failed",
                        "error": str(exc),
                        "metrics": {},
                    }
                )


        # Aggregate
        aggregate_metrics = self._aggregate_metrics( query_results)

        successful_queries = sum(1 for result in query_results
            if result["status"] == "success")

        failed_queries = sum(1 for result in query_results
            if result["status"] == "failed")

        accuracy = {} 
        for k in k_values: 
            metric_name = f"Document_Hit@{k}" 

            if metric_name in aggregate_metrics: 
                accuracy[f"Document_Accuracy@{k}"] = (aggregate_metrics[metric_name] * 100)

        return {
            "total_queries": len(dataset),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "metrics": aggregate_metrics,
            "accuracy_percentage": accuracy,
            "queries": query_results,
        }


    # AGGREGATION
    def _aggregate_metrics(self, query_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate the mean value of each metric across
        successful queries.
        """

        successful_results = [result for result in query_results
            if result["status"] == "success"]

        if not successful_results:
            return {}

        # Find every metric that occured
        metric_names = set()

        for result in successful_results: metric_names.update(
                result.get("metrics", {}).keys())

        # Calculate mean
        aggregated = {}

        for metric_name in sorted(metric_names):

            values = [
                result["metrics"][metric_name]
                for result in successful_results
                if metric_name in result.get("metrics", {})
            ]

            if not values:
                continue

            aggregated[metric_name] = (sum(values) / len(values))

        return aggregated