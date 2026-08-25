import logging
from typing import Any, Dict, List

from tools.BM25_retriever_tool import BM25RetrieverTool
from tools.retriever_tool import RetrieverTool
from tools.vector_retriever_tool import VectorRetrieverTool


logger = logging.getLogger(__name__)


class HybridRetrieverTool(RetrieverTool):
    """
    Hybrid retrieval using:

        1. Vector similarity retrieval
        2. BM25 lexical retrieval
        3. BM25 score normalization
        4. Reciprocal Rank Fusion (RRF)
        5. Hybrid confidence calculation

    Scores:

        vector_score
            Original vector similarity score from Qdrant.

        bm25_score
            Raw BM25 score.

        bm25_normalized
            BM25 score normalized to [0, 1] within
            the current BM25 result set.

        fusion_score
            RRF ranking score.

        confidence_score
            Combined confidence based on vector and
            normalized BM25 scores.
    """

    name = "Hybrid Retriever"

    def __init__(
        self,
        vector_tool: VectorRetrieverTool,
        bm25_retriever: BM25RetrieverTool,
        rrf_k: int = 60,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.vector_tool = vector_tool
        self.bm25_retriever = bm25_retriever

        self.rrf_k = rrf_k

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # Make sure weights are valid.
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError(
                "vector_weight and bm25_weight must be >= 0"
            )

        if vector_weight + bm25_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0"
            )

    # ---------------------------------------------------------
    # BM25 NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_bm25_scores(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize raw BM25 scores into the range [0, 1].

        Normalization is performed relative to the current
        BM25 result set.

        Example:

            Raw BM25:
                8.0
                4.0
                2.0

            Normalized:
                1.0
                0.5
                0.25
        """

        if not results:
            return results

        max_score = max(
            float(result.get("bm25_score", 0.0))
            for result in results
        )

        # If all BM25 scores are zero.
        if max_score <= 0:
            for result in results:
                result["bm25_normalized"] = 0.0

            return results

        for result in results:

            bm25_score = float(
                result.get("bm25_score", 0.0)
            )

            normalized_score = bm25_score / max_score

            # Safety clamp.
            normalized_score = max(
                0.0,
                min(1.0, normalized_score)
            )

            result["bm25_normalized"] = normalized_score

        return results

    # ---------------------------------------------------------
    # DOCUMENT ID
    # ---------------------------------------------------------

    def _get_document_id(
        self,
        result: Dict[str, Any],
    ) -> str:
        """
        Get the stable chunk/document ID.

        Prefer metadata.chunk_id because the same chunk ID
        should exist in both vector and BM25 stores.
        """

        metadata = result.get("metadata", {}) or {}

        return str(
            metadata.get(
                "chunk_id",
                result.get("id")
            )
        )

    # ---------------------------------------------------------
    # RETRIEVE
    # ---------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5,) -> List[Dict[str, Any]]:

        logger.info(
            "[HybridRetrieverTool] Starting hybrid retrieval "
            "query='%s', top_k=%d",
            query,
            top_k,
        )

        # -----------------------------------------------------
        # 1. VECTOR RETRIEVAL
        # -----------------------------------------------------

        vector_results = self.vector_tool.retrieve(
            query=query,
            top_k=top_k,
        )

        logger.info(
            "[HybridRetrieverTool] Vector results=%d",
            len(vector_results),
        )

        # -----------------------------------------------------
        # 2. BM25 RETRIEVAL
        # -----------------------------------------------------

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        logger.info(
            "[HybridRetrieverTool] BM25 results=%d",
            len(bm25_results),
        )

        # -----------------------------------------------------
        # 3. NORMALIZE BM25 SCORES
        # -----------------------------------------------------

        bm25_results = self._normalize_bm25_scores(
            bm25_results
        )

        # -----------------------------------------------------
        # 4. FUSE RESULTS
        # -----------------------------------------------------

        fused: Dict[str, Dict[str, Any]] = {}

        # =====================================================
        # VECTOR RESULTS
        # =====================================================

        for rank, result in enumerate(
            vector_results,
            start=1,
        ):

            document_id = self._get_document_id(result)

            # VectorRetriever currently returns
            # "similarity_score".
            #
            # Fall back to "score" for compatibility.
            vector_score = float(
                result.get(
                    "similarity_score",
                    result.get("score", 0.0),
                )
            )

            if document_id not in fused:

                fused[document_id] = {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),

                    "vector_score": vector_score,

                    "bm25_score": 0.0,

                    "bm25_normalized": 0.0,

                    "vector_rank": rank,

                    "bm25_rank": None,

                    "fusion_score": 0.0,
                }

            else:

                # In case the document was already added
                # from BM25 retrieval.
                fused[document_id]["vector_score"] = (
                    vector_score
                )

                fused[document_id]["vector_rank"] = rank

            # RRF contribution from vector search.
            fused[document_id]["fusion_score"] += (
                1.0 / (self.rrf_k + rank)
            )

        # =====================================================
        # BM25 RESULTS
        # =====================================================

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            document_id = self._get_document_id(result)

            bm25_score = float(
                result.get(
                    "bm25_score",
                    result.get("score", 0.0),
                )
            )

            bm25_normalized = float(
                result.get(
                    "bm25_normalized",
                    0.0,
                )
            )

            if document_id not in fused:

                fused[document_id] = {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result.get(
                        "metadata",
                        {},
                    ),

                    "vector_score": 0.0,

                    "bm25_score": bm25_score,

                    "bm25_normalized": bm25_normalized,

                    "vector_rank": None,

                    "bm25_rank": rank,

                    "fusion_score": 0.0,
                }

            else:

                # Document already exists because it was
                # retrieved by vector search.

                fused[document_id]["bm25_score"] = (
                    bm25_score
                )

                fused[document_id]["bm25_normalized"] = (
                    bm25_normalized
                )

                fused[document_id]["bm25_rank"] = rank

            # RRF contribution from BM25.
            fused[document_id]["fusion_score"] += (
                1.0 / (self.rrf_k + rank)
            )

        # -----------------------------------------------------
        # 5. SORT BY RRF FUSION SCORE
        # -----------------------------------------------------

        results = sorted(
            fused.values(),
            key=lambda x: x["fusion_score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # 6. BUILD FINAL RESULTS
        # -----------------------------------------------------

        final_results: List[Dict[str, Any]] = []

        for rank, result in enumerate(
            results[:top_k],
            start=1,
        ):

            vector_score = float(
                result.get(
                    "vector_score",
                    0.0,
                )
            )

            bm25_normalized = float(
                result.get(
                    "bm25_normalized",
                    0.0,
                )
            )

            # -------------------------------------------------
            # HYBRID CONFIDENCE
            # -------------------------------------------------
            #
            # Vector retrieval is given 70% weight.
            # BM25 retrieval is given 30% weight.
            #
            # Both values are now in [0, 1].
            #

            confidence_score = (
                self.vector_weight * vector_score
                +
                self.bm25_weight * bm25_normalized
            )

            # Safety clamp.
            confidence_score = max(
                0.0,
                min(1.0, confidence_score),
            )

            final_results.append(
                {
                    "id": result["metadata"].get(
                        "chunk_id",
                        result["id"],
                    ),

                    "content": result["content"],

                    "metadata": result["metadata"],

                    # Original vector similarity.
                    "vector_score": vector_score,

                    # Raw BM25 score.
                    "bm25_score": result.get(
                        "bm25_score",
                        0.0,
                    ),

                    # Normalized BM25 score.
                    "bm25_normalized": bm25_normalized,

                    # RRF score.
                    "fusion_score": result[
                        "fusion_score"
                    ],

                    # Final confidence.
                    "confidence_score": confidence_score,

                    # Keep "score" for compatibility
                    # with RetrievalAgent.
                    "score": confidence_score,

                    "rank": rank,

                    "vector_rank": result.get(
                        "vector_rank"
                    ),

                    "bm25_rank": result.get(
                        "bm25_rank"
                    ),
                }
            )

        # -----------------------------------------------------
        # 7. LOGGING
        # -----------------------------------------------------

        logger.info(
            "[HybridRetrieverTool] "
            "vector=%d bm25=%d final=%d",
            len(vector_results),
            len(bm25_results),
            len(final_results),
        )

        for result in final_results:

            logger.debug(
                "[HybridRetrieverTool] "
                "rank=%d vector=%.4f bm25=%.4f "
                "bm25_normalized=%.4f fusion=%.4f "
                "confidence=%.4f",
                result["rank"],
                result["vector_score"],
                result["bm25_score"],
                result["bm25_normalized"],
                result["fusion_score"],
                result["confidence_score"],
            )

        return final_results