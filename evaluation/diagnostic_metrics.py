from typing import Any, Dict, List, Optional, Set


class DiagnosticMetrics:
    """
        A class for calculating diagnostic metrics for retrieval results.

        Retrieval Diagnostocs -
        - DuplicateRate@k
        - MetadataCompleteness@k
        - EmptyResultRate@k
        - DocumentConcentration@k

        Ground Truth Diagnostics -
        - RelevantDocumentRank
        - RelevantChunkRank
        - DocumentRank@k
        - ChunkRank@k

        Similarity Score Diagnostics -
        - TopScore
        - RelevantScore
        - ScoreGap
        - ScoreDecay

        Severity -
        - NONE
        - LOW
        - MEDIUM
        - HIGH
        - CRITICAL

    """


    # Severity levels for diagnostic metrics
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


    # DuplicateRate@k
    @staticmethod
    def duplicate_rate_at_k(retrieved_ids: List[Optional[str]], k: int) -> float:
        """
            Calculate the proportion of duplicate IDs within top-k results.

            Example:
                [A, B, A, C, B] with k=5
                duplicates = A + B = 2
                duplicate rate = 2 / 5 = 0.4
        """

        if k <= 0 or not retrieved_ids:
            return 0.0

        retrieved = retrieved_ids[:k]

        if not retrieved:
            return 0.0

        seen = set()
        duplicates = 0

        for item_id in retrieved:
            if item_id is None:
                continue

            if item_id in seen:
                duplicates += 1
            else:
                seen.add(item_id)

        return duplicates / len(retrieved)


    # MetadataCompleteness@k
    @staticmethod
    def metadata_completeness_at_k(retrieved_results: List[Dict[str, Any]], k: int) -> float:
        """
        Calculate the proportion of top-k results with required metadata.

        Required identity fields:
            - document_id
            - chunk_id
        """

        if k <= 0 or not retrieved_results:
            return 0.0

        retrieved = retrieved_results[:k]

        if not retrieved:
            return 0.0

        complete_count = 0

        for result in retrieved:
            document_id = result.get("document_id")
            chunk_id = result.get("chunk_id")

            if document_id is not None and chunk_id is not None:
                complete_count += 1

        return complete_count / len(retrieved)


    # EmptyResultRate@k
    @staticmethod
    def empty_retrieval_at_k(retrieved_results: List[Dict[str, Any]], k: int) -> float:
        """
        Return 1.0 if there are no results within top-k, otherwise 0.0.
        """

        if k <= 0:
            return 0.0

        return float(len(retrieved_results[:k]) == 0)


    # DocumentCoverage@k
    @staticmethod
    def document_concentration_at_k(retrieved_document_ids: List[Optional[str]], 
                                    k: int) -> float:
        """
        Measure how concentrated the top-k results are around one document.

        Example:
            [A, A, A, B, C] -> 3 / 5 = 0.6

        A high value means one document dominates the retrieved results.
        """

        if k <= 0 or not retrieved_document_ids:
            return 0.0

        retrieved = retrieved_document_ids[:k]

        valid_ids = [
            document_id
            for document_id in retrieved
            if document_id is not None
        ]

        if not valid_ids:
            return 0.0

        counts = {}

        for document_id in valid_ids:
            counts[document_id] = counts.get(document_id, 0) + 1

        max_count = max(counts.values())

        return max_count / len(retrieved)

    @staticmethod
    def first_relevant_rank(retrieved_results: List[Dict[str, Any]], relevant_ids: Set[str], 
                            id_field: str) -> Optional[int]:
        """
        Return the first 1-based rank containing a relevant ID.

        Returns None when no relevant item is found.
        """

        if not retrieved_results or not relevant_ids:
            return None

        for index, result in enumerate(retrieved_results, start=1):
            item_id = result.get(id_field)

            if item_id is not None and str(item_id) in relevant_ids:
                return index

        return None


    @staticmethod
    def first_relevant_score(retrieved_results: List[Dict[str, Any]], relevant_ids: Set[str],
                             id_field: str) -> Optional[float]:
        """
            Return the similarity score of the first relevant result.
        """

        if not retrieved_results or not relevant_ids:
            return None

        for result in retrieved_results:
            item_id = result.get(id_field)

            if item_id is not None and str(item_id) in relevant_ids:
                score = result.get("similarity_score")

                if score is not None:
                    return float(score)

        return None


    @staticmethod
    def top_similarity_score(retrieved_results: List[Dict[str, Any]]) -> Optional[float]:
        """
            Return the similarity score of the highest-ranked result.
        """

        if not retrieved_results:
            return None

        score = retrieved_results[0].get("similarity_score")

        if score is None:
            return None

        return float(score)


    @staticmethod
    def score_gap(top_score: Optional[float], 
                  relevant_score: Optional[float]) -> Optional[float]:
        """
            Calculate: top_score - relevant_score

            A small gap means the relevant result was close to the top result.
            A large gap indicates stronger score separation.
        """

        if top_score is None or relevant_score is None:
            return None

        return float(top_score - relevant_score)


    # ScoreDecay
    @staticmethod
    def score_decay(retrieved_results: List[Dict[str, Any]], 
                    k: Optional[int] = None) -> Optional[float]:
        """
            Calculate normalized similarity-score decay from rank 1 to rank K.
            
            Formula: (top_score - kth_score) / abs(top_score)
        """
        if not retrieved_results:
            return None

        if k is None:
            k = len(retrieved_results)

        if k <= 0:
            return None

        top_k = retrieved_results[:k]

        if not top_k:
            return None

        top_score = top_k[0].get("similarity_score")
        kth_score = top_k[-1].get("similarity_score")

        if top_score is None or kth_score is None:
            return None

        if top_score == 0:
            return None

        return (float(top_score) - float(kth_score)) / abs(top_score)
    

    # SeverityFromRate
    @staticmethod
    def severity_from_rate(rate: Optional[float], low_threshold: float, 
                           medium_threshold: float, high_threshold: float, 
                           critical_threshold: float) -> str:
        """
            Convert a diagnostic rate into a severity level.

            Thresholds are supplied by the caller rather than hard-coded
            because severity is application-specific.
        """
        if rate is None:
            return DiagnosticMetrics.NONE

        rate = float(rate)

        if rate < low_threshold:
            return DiagnosticMetrics.NONE

        if rate < medium_threshold:
            return DiagnosticMetrics.LOW

        if rate < high_threshold:
            return DiagnosticMetrics.MEDIUM

        if rate < critical_threshold:
            return DiagnosticMetrics.HIGH

        return DiagnosticMetrics.CRITICAL