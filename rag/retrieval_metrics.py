class RetrievalMetrics:
    """
    Traditional information-retrieval metrics.

    These metrics operate on ranked IDs and
    ground-truth relevance information.
    """

    # =====================================================
    # Precision@K
    # =====================================================

    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k: int) -> float:
        """
            Of the retrieved results in the top-k,
            what fraction are relevant?

            Formula: relevant items in top-k / retrieved items in top-k
        """

        if k <= 0:
            return 0.0

        if not retrieved_ids:
            return 0.0

        if not relevant_ids:
            return 0.0

        retrieved = list(dict.fromkeys(retrieved_ids))[:k]

        relevant_hits = sum(
            1
            for item_id in retrieved
            if item_id in relevant_ids
        )

        return relevant_hits / len(retrieved)

    # =====================================================
    # Recall@K
    # =====================================================

    @staticmethod
    def recall_at_k(
        retrieved_ids,
        relevant_ids,
        k: int
    ) -> float:
        """
        Of all relevant items, what fraction were
        retrieved in the top-k?

        Formula:

            relevant items retrieved in top-k
            ---------------------------------
                 total relevant items
        """

        if k <= 0:
            return 0.0

        if not retrieved_ids:
            return 0.0

        if not relevant_ids:
            return 0.0

        retrieved = list(dict.fromkeys(retrieved_ids))[:k]

        relevant_hits = sum(
            1
            for item_id in retrieved
            if item_id in relevant_ids
        )

        return relevant_hits / len(relevant_ids)

    # =====================================================
    # Hit@K
    # =====================================================

    @staticmethod
    def hit_at_k(
        retrieved_ids,
        relevant_ids,
        k: int
    ) -> float:
        """
        Returns 1.0 if at least one relevant item
        appears in the top-k results, otherwise 0.0.

        Useful for measuring whether the correct
        document/chunk was found at all.
        """

        if k <= 0:
            return 0.0

        if not retrieved_ids or not relevant_ids:
            return 0.0

        retrieved = retrieved_ids[:k]

        return float(
            any(
                item_id in relevant_ids
                for item_id in retrieved
            )
        )
