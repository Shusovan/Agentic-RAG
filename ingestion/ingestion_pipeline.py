import logging
from typing import List

from ingestion.metadata_extractor import MetadataExtractor
from rag.chunking import ChunkingPipeline


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Handles end-to-end document ingestion

    Steps:
    1. Extract metadata
    2. Chunk documents
    3. Generate embeddings
    4. Store in vector DB
    """

    def __init__(self, vector_store, embedding_pipeline):
        
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

        self.metadata_extractor = MetadataExtractor()
        self.chunking_pipeline = ChunkingPipeline()

    def process_documents(self, documents: List):
        """
        Runs the full ingestion pipeline
        """

        try:

            logger.info("Starting ingestion pipeline")

            # Step 1 — Metadata enrichment
            documents = self.metadata_extractor.extract(documents)

            # Step 2 — Chunking
            chunked_docs = []

            for doc in documents:

                chunks = self.chunking_pipeline.split_text(doc.page_content)

                for chunk in chunks:

                    new_doc = type(doc)(page_content=chunk.page_content, metadata=doc.metadata)

                    chunked_docs.append(new_doc)

            logger.info(f"Created {len(chunked_docs)} chunks")

            # Step 3 — Prepare texts for embedding
            texts = [doc.page_content for doc in chunked_docs]

            embeddings = self.embedding_pipeline.embed_documents(texts)

            # Step 4 — Store in vector DB
            self.vector_store.add_documents(documents=chunked_docs, embeddings=embeddings)

            logger.info("Ingestion pipeline completed successfully")

        except Exception as e:

            logger.exception("Ingestion pipeline failed")

            raise ValueError(f"Ingestion pipeline failed: {e}")
