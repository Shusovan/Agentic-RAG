from typing import List, Optional


class DiagnosticMetrics:

    # DuplicateRate@k
    @staticmethod
    def duplicate_rate_at_k(retrieved_ids: List[Optional[str]], k: int) -> float:
        """
            Measures the fraction of top-k retrieval results 
            that duplicate an ID already seen earlier.
        """

        if k < 0:
            return 0.0

        retrieved = retrieved_ids[:k]

        if retrieved is None or len(retrieved) == 0:
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
    def metadata_completeness_at_k(retrieved_metadata: List[dict], k: int) -> float:
        """
            Measures the fraction of top-k retrieval results 
            that have complete metadata (i.e., all required fields are present).
        """

        if k < 0:
            return 0.0

        retrieved = retrieved_metadata[:k]

        if retrieved is None or len(retrieved) == 0:
            return 0.0

        complete_count = sum( 1 
                             for result in retrieved 
                             if result.get("document_id") is not None 
                             and result.get("chunk_id") is not None ) 

        return complete_count / len(retrieved)