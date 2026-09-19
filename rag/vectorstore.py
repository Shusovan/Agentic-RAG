import logging
import os
import threading
import uuid

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)

from typing import Any, Dict, List


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Handles storing and retrieving embeddings from Qdrant.
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

        # -------------------------------------------------
        # Protect Qdrant Local writes
        # -------------------------------------------------

        self._write_lock = threading.Lock()

        self._initialize_qdrant()


    # initialize Qdrant client and collection
    def _initialize_qdrant(self):

        try:

            os.makedirs(self.persist_directory, exist_ok=True)

            logger.info("QDRANT DB PATH: %s", os.path.abspath(self.persist_directory))

            self.client = QdrantClient(path=self.persist_directory)

            collections = (self.client.get_collections().collections)

            collection_names = [collection.name
                for collection in collections]

            if self.collection_name not in collection_names:
                logger.info("Creating collection: %s", self.collection_name)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )

            logger.info("Collection '%s' ready.",self.collection_name)

        except Exception as exc:
            logger.exception("Failed to initialize Qdrant")
            raise ValueError(f"Failed to initialize Qdrant: {exc}") from exc


    # add documents to Qdrant
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):

        if not documents:
            logger.warning("No documents supplied to vector store")
            return

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        logger.info("Adding %d documents to vector store", len(documents))

        points = []

        for document, embedding in zip(documents, embeddings):

            # qdrant internal point ID
            point_id = uuid.uuid4().hex

            # document ID (application level ID)
            document_id = document.metadata.get("document_id")

            # chunk ID
            chunk_id = document.metadata.get("chunk_id")

            # source
            source = document.metadata.get("source")

            if not document_id:
                raise ValueError("document has no document_id")

            if not chunk_id:
                raise ValueError("document has no chunk_id")

            payload = {
                "text": document.page_content,
                "metadata": {
                    **dict(document.metadata),

                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "source": source
                },
            }

            points.append(PointStruct(id=point_id, vector=embedding.tolist(), 
                                      payload=payload))

        if not points:
            logger.warning("No valid points to store")
            return

        # Qdrant Local is SQLite-backed.
        # Serialize writes.
        with self._write_lock:
            try:
                logger.info("Writing %d points to Qdrant",len(points))

                logger.info("QDRANT WRITE: collection=%s path=%s points=%d",
                self.collection_name,
                os.path.abspath(self.persist_directory),
                len(points))

                for point in points[:3]:
                    logger.info("QDRANT POINT: id=%s payload=%s",
                        point.id,
                        point.payload)

                self.client.upsert(collection_name=self.collection_name, points=points)

                logger.info("%d documents stored successfully", len(points))

            except Exception as exc:
                logger.exception("Failed to store documents in Qdrant")
                raise ValueError(f"Failed to store documents in Qdrant: "
                    f"{exc}") from exc


    # reset the collection (delete and recreate)
    def reset_collection(self):

        with self._write_lock:
            try:
                if self.client.collection_exists(self.collection_name):
                    logger.warning("Deleting Qdrant collection: %s",
                        self.collection_name)

                    self.client.delete_collection(collection_name=self.collection_name)

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )

                logger.info("Collection '%s' reset successfully",
                    self.collection_name)

            except Exception as exc:
                logger.exception("Failed to reset Qdrant collection")

                raise ValueError(f"Failed to reset Qdrant collection: {exc}") from exc


    # query Qdrant for relevant documents
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:

        logger.info(
            "Searching vector DB with top_k=%d",
            top_k
        )

        try:

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k
            )

            formatted_results = []

            for point in response.points:

                payload = point.payload or {}

                metadata = payload.get("metadata", {})

                formatted_results.append(
                    {
                        "id": str(point.id),
                        "document_id": metadata.get("document_id"),
                        "chunk_id": metadata.get("chunk_id"),
                        "content": payload.get("text", ""),
                        "metadata": metadata,
                        "score": point.score
                    })

            return formatted_results

        except Exception as exc:
            logger.exception("Vector search failed")
            raise ValueError(f"Vector search failed: {exc}") from exc


    # delete by source
    def delete_by_source(self, source_name: str):

        try:
            logger.info("Deleting embeddings for source: %s", source_name)

            with self._write_lock:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source",
                                match=models.MatchValue(
                                    value=source_name
                                )
                            )
                        ]
                    )
                )

            logger.info("Embeddings deleted for source: %s", source_name)

        except Exception as exc:
            logger.exception("Failed to delete embeddings for %s", source_name)
            raise ValueError(f"Failed to delete embeddings: {exc}") from exc