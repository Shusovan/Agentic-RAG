from typing import Dict, List, Set, Any, Optional

from evaluation.diagnostic_metrics import DiagnosticMetrics


class DiagnosticEvaluator:
    """
    Diagnostic evaluator for retrieval results.

    This evaluator does not perform retrieval.

    It consumes the output of RetrievalEvaluator and produces:
        - Retrieval diagnostic measurements
        - Ground-truth ranking diagnostics
        - Similarity-score diagnostics
        - Objective retrieval issues

    Severity classification is intentionally not implemented yet.
    Thresholds will be introduced later after sufficient evaluation
    data is available.
    """

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
    ):
        """
        Initialize DiagnosticEvaluator.

        Args:
            k_values:
                K values to evaluate.
                Example: [1, 3, 5, 10]
        """

        self.k_values = sorted(
            set(k_values or [1, 3, 5, 10])
        )

    # ================================================================
    # Public API
    # ================================================================

    def evaluate_query(
        self,
        retrieval_query_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Diagnose a single retrieval-evaluation query.

        Expected input is one query result produced by
        RetrievalEvaluator.
        """

        if not retrieval_query_result:
            raise ValueError(
                "retrieval_query_result cannot be empty"
            )

        query = retrieval_query_result.get(
            "query",
            "",
        )

        ground_truth = retrieval_query_result.get(
            "ground_truth",
            {},
        )

        retrieved_data = self._extract_retrieved_data(
            retrieval_query_result
        )

        # ------------------------------------------------------------
        # Extract IDs
        # ------------------------------------------------------------

        retrieved_document_ids = [
            result.get("document_id")
            for result in retrieved_data
        ]

        retrieved_chunk_ids = [
            result.get("chunk_id")
            for result in retrieved_data
        ]

        relevant_document_ids = (
            self._extract_relevant_document_ids(
                ground_truth
            )
        )

        relevant_chunk_ids = (
            self._extract_relevant_chunk_ids(
                ground_truth
            )
        )

        # ------------------------------------------------------------
        # Calculate diagnostic metrics for each K
        # ------------------------------------------------------------

        diagnostics = {}

        for k in self.k_values:

            duplicate_rate = (
                DiagnosticMetrics.duplicate_rate_at_k(
                    retrieved_chunk_ids,
                    k,
                )
            )

            metadata_completeness = (
                DiagnosticMetrics.metadata_completeness_at_k(
                    retrieved_data,
                    k,
                )
            )

            empty_result = (
                DiagnosticMetrics.empty_retrieval_at_k(
                    retrieved_data,
                    k,
                )
            )

            document_concentration = (
                DiagnosticMetrics.document_concentration_at_k(
                    retrieved_document_ids,
                    k,
                )
            )

            score_decay = (
                DiagnosticMetrics.score_decay(
                    retrieved_data,
                    k,
                )
            )

            diagnostics[f"@{k}"] = {
                "DuplicateRate": duplicate_rate,
                "MetadataCompleteness": (
                    metadata_completeness
                ),
                "EmptyResult": empty_result,
                "DocumentConcentration": (
                    document_concentration
                ),
                "ScoreDecay": score_decay,
            }

        # ------------------------------------------------------------
        # Ground-truth diagnostics
        # ------------------------------------------------------------

        document_rank = (
            DiagnosticMetrics.first_relevant_rank(
                retrieved_data,
                relevant_document_ids,
                "document_id",
            )
        )

        chunk_rank = (
            DiagnosticMetrics.first_relevant_rank(
                retrieved_data,
                relevant_chunk_ids,
                "chunk_id",
            )
        )

        document_score = (
            DiagnosticMetrics.first_relevant_score(
                retrieved_data,
                relevant_document_ids,
                "document_id",
            )
        )

        chunk_score = (
            DiagnosticMetrics.first_relevant_score(
                retrieved_data,
                relevant_chunk_ids,
                "chunk_id",
            )
        )

        top_score = (
            DiagnosticMetrics.top_similarity_score(
                retrieved_data
            )
        )

        document_score_gap = (
            DiagnosticMetrics.score_gap(
                top_score,
                document_score,
            )
        )

        chunk_score_gap = (
            DiagnosticMetrics.score_gap(
                top_score,
                chunk_score,
            )
        )

        # ------------------------------------------------------------
        # Objective diagnosis
        # ------------------------------------------------------------

        issues = self._diagnose_issues(
            retrieved_data=retrieved_data,
            relevant_document_ids=(
                relevant_document_ids
            ),
            relevant_chunk_ids=(
                relevant_chunk_ids
            ),
            diagnostics=diagnostics,
            document_rank=document_rank,
            chunk_rank=chunk_rank,
        )

        # ------------------------------------------------------------
        # Final result
        # ------------------------------------------------------------

        return {
            "query": query,

            "ground_truth": ground_truth,

            "retrieval_summary": {
                "total_results": len(
                    retrieved_data
                ),
                "top_score": top_score,
                "document_rank": document_rank,
                "chunk_rank": chunk_rank,
                "document_score": document_score,
                "chunk_score": chunk_score,
                "document_score_gap": (
                    document_score_gap
                ),
                "chunk_score_gap": (
                    chunk_score_gap
                ),
            },

            "diagnostics": diagnostics,

            "issues": issues,
        }

    # ================================================================
    # Dataset evaluation
    # ================================================================

    def evaluate_dataset(
        self,
        retrieval_evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Diagnose all query results produced by RetrievalEvaluator.
        """

        if not retrieval_evaluation:
            raise ValueError(
                "retrieval_evaluation cannot be empty"
            )

        queries = retrieval_evaluation.get(
            "queries",
            [],
        )

        results = []

        successful = 0
        failed = 0

        for query_result in queries:

            try:

                diagnostic_result = (
                    self.evaluate_query(
                        query_result
                    )
                )

                results.append(
                    diagnostic_result
                )

                successful += 1

            except Exception as exc:

                failed += 1

                results.append({
                    "query": query_result.get(
                        "query",
                        "",
                    ),
                    "error": str(exc),
                })

        summary = self._build_summary(
            results
        )

        return {
            "evaluation_type": "diagnostic",

            "total_queries": len(
                queries
            ),

            "successful_queries": successful,

            "failed_queries": failed,

            "summary": summary,

            "queries": results,
        }

    # ================================================================
    # Data extraction
    # ================================================================

    @staticmethod
    def _extract_retrieved_data(
        retrieval_query_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract retrieved result records.

        Supports:

            "retrieved_data": [...]

        and:

            "retrieved": [...]

        and:

            "retrieved": {
                "retrieved_data": [...]
            }
        """

        retrieved_data = (
            retrieval_query_result.get(
                "retrieved_data"
            )
        )

        if retrieved_data is not None:
            return retrieved_data

        retrieved = (
            retrieval_query_result.get(
                "retrieved",
                [],
            )
        )

        if isinstance(retrieved, list):
            return retrieved

        if isinstance(retrieved, dict):

            return retrieved.get(
                "retrieved_data",
                [],
            )

        return []

    @staticmethod
    def _extract_relevant_document_ids(
        ground_truth: Dict[str, Any],
    ) -> Set[str]:
        """
        Extract document-level ground truth IDs.
        """

        document_ids = ground_truth.get(
            "document_ids"
        )

        if document_ids:
            return set(document_ids)

        document_id = ground_truth.get(
            "document_id"
        )

        if document_id:
            return {document_id}

        return set()

    @staticmethod
    def _extract_relevant_chunk_ids(
        ground_truth: Dict[str, Any],
    ) -> Set[str]:
        """
        Extract chunk-level ground truth IDs.
        """

        chunk_ids = ground_truth.get(
            "chunk_ids"
        )

        if chunk_ids:
            return set(chunk_ids)

        chunk_id = ground_truth.get(
            "chunk_id"
        )

        if chunk_id:
            return {chunk_id}

        return set()

    # ================================================================
    # Objective issue detection
    # ================================================================

    def _diagnose_issues(
        self,
        retrieved_data: List[Dict[str, Any]],
        relevant_document_ids: Set[str],
        relevant_chunk_ids: Set[str],
        diagnostics: Dict[str, Any],
        document_rank: Optional[int],
        chunk_rank: Optional[int],
    ) -> List[str]:
        """
        Identify objective retrieval issues.

        No heuristic severity thresholds are used here.
        """

        issues = []

        # ------------------------------------------------------------
        # 1. Empty retrieval
        # ------------------------------------------------------------

        if not retrieved_data:

            issues.append(
                "EMPTY_RETRIEVAL"
            )

            return issues

        # ------------------------------------------------------------
        # 2. Document not retrieved
        # ------------------------------------------------------------

        if (
            relevant_document_ids
            and document_rank is None
        ):

            issues.append(
                "DOCUMENT_NOT_RETRIEVED"
            )

        # ------------------------------------------------------------
        # 3. Chunk not retrieved
        # ------------------------------------------------------------

        if (
            relevant_chunk_ids
            and chunk_rank is None
        ):

            if document_rank is not None:

                issues.append(
                    "DOCUMENT_FOUND_CHUNK_MISSED"
                )

            else:

                issues.append(
                    "CHUNK_NOT_RETRIEVED"
                )

        # ------------------------------------------------------------
        # 4. Metadata completeness
        # ------------------------------------------------------------

        if self._has_incomplete_metadata(
            retrieved_data
        ):

            issues.append(
                "INCOMPLETE_METADATA"
            )

        # ------------------------------------------------------------
        # 5. Duplicate retrieval records
        # ------------------------------------------------------------

        if self._has_duplicate_chunks(
            retrieved_data
        ):

            issues.append(
                "DUPLICATE_RETRIEVAL"
            )

        return issues

    # ================================================================
    # Structural checks
    # ================================================================

    @staticmethod
    def _has_incomplete_metadata(
        retrieved_data: List[Dict[str, Any]],
    ) -> bool:
        """
        Detect missing document_id or chunk_id.

        This is an objective structural condition rather than
        a threshold-based judgment.
        """

        for result in retrieved_data:

            if (
                result.get("document_id") is None
                or result.get("chunk_id") is None
            ):

                return True

        return False

    @staticmethod
    def _has_duplicate_chunks(
        retrieved_data: List[Dict[str, Any]],
    ) -> bool:
        """
        Detect exact duplicate chunk IDs.

        None values are ignored.
        """

        seen = set()

        for result in retrieved_data:

            chunk_id = result.get(
                "chunk_id"
            )

            if chunk_id is None:
                continue

            if chunk_id in seen:
                return True

            seen.add(chunk_id)

        return False

    # ================================================================
    # Summary
    # ================================================================

    @staticmethod
    def _build_summary(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Aggregate objective diagnostic findings.
        """

        valid_results = [
            result
            for result in results
            if "error" not in result
        ]

        issue_counts = {}

        for result in valid_results:

            for issue in result.get(
                "issues",
                [],
            ):

                issue_counts[issue] = (
                    issue_counts.get(
                        issue,
                        0,
                    )
                    + 1
                )

        return {
            "issue_counts": issue_counts,
        }