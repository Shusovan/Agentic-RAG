import hashlib
import logging
import os
import uuid

import numpy as np
from qdrant_client import QdrantClient

from typing import Any, Dict, List

from qdrant_client import models
from qdrant_client.models import Distance, PointStruct, VectorParams


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Handles storing and retrieving embeddings from Qdrant
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./qdrant_db",
        embedding_dim: int = 384
    ):

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim

        self.client = None

        self._initialize_qdrant()


    def _initialize_qdrant(self):
        """Initialize Qdrant client and collection"""

        try:

            os.makedirs(self.persist_directory, exist_ok=True)

            logger.info(f"Initializing Qdrant client")

            self.client = QdrantClient(path=self.persist_directory)

            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:

                logger.info(f"Creating collection: {self.collection_name}")

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )

            logger.info(f"Collection '{self.collection_name}' ready.")

        except Exception as e:
            raise ValueError(f"Failed to initialize Qdrant: {e}")


    # Generating Chunk ID
    @staticmethod
    def _generate_chunk_id(document: Any, index: int) -> str:

        metadata = dict(getattr(document, "metadata", {}) or {})

        source = metadata.get("source", "unknown")

        raw = (f"{source}|" f"{index}|" f"{document.page_content}")

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Store document embeddings in Qdrant
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        logger.info(f"Adding {len(documents)} documents to vector store")

        points = []

        for index, (doc, embedding) in enumerate(zip(documents, embeddings)):

            metadata = dict(getattr(doc, "metadata", {}) or {})

            chunk_id = metadata.get("chunk_id") or self._generate_chunk_id(doc, index)

            logger.info(f"Document index={index} | chunk_id={chunk_id} | "
                        f"embedding_dim={len(embedding)}")

            metadata["chunk_id"] = chunk_id

            # Qdrant requires a valid UUID or integer for the point ID.
            # Generate a deterministic UUID from the chunk ID.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

            payload = {"text": doc.page_content, "metadata": metadata,}

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)

            logger.info(f"Successfully added {len(points)} documents to vector store")

        except Exception as e:
            raise ValueError(f"Failed to add documents to vector store: {e}")

    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:

        logger.info(f"Searching vector DB with top_k={top_k}")

        try:

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
            )

            formatted_results = []

            for point in response.points:

                metadata = (
                    point.payload.get("metadata", {})
                    if point.payload else {}
                )

                formatted_results.append({
                    "id": point.id,
                    "content": point.payload["text"],
                    "metadata": metadata,
                    "score": float(point.score),
                })

            return formatted_results

        except Exception as e:
            raise ValueError(f"Vector search failed: {e}")
        

    def delete_by_source(self, source_name: str):
        """
        Delete all embeddings belonging to a specific source document.
        Used when a document is updated or deleted in Corpora folder.
        """

        try:

            logger.info(f"Deleting embeddings for source: {source_name}")

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source_name)
                        )
                    ]
                )
            )

            logger.info(f"Embeddings deleted for source: {source_name}")

        except Exception as e:

            logger.error(f"Failed to delete embeddings for {source_name}: {e}")

            raise ValueError(f"Failed to delete embeddings: {e}")