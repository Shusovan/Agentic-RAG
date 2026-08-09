import logging
from typing import List

from langchain_core.documents import Document

from ingestion.metadata_extractor import MetadataExtractor
from rag.chunking import ChunkingPipeline


logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the complete document ingestion workflow.

    Pipeline:
        1. Metadata enrichment
        2. Chunking
        3. Embedding generation
        4. Vector store ingestion
    """

    def __init__(
        self,
        vector_store,
        embedding_pipeline,
        metadata_extractor: MetadataExtractor | None = None,
        chunking_pipeline: ChunkingPipeline | None = None,
    ):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

        # Allow dependency injection for testing/extensibility
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.chunking_pipeline = chunking_pipeline or ChunkingPipeline()

    def process_documents(self, documents: List[Document]) -> None:
        """
        Execute the complete ingestion pipeline.

        Args:
            documents: List of LangChain Document objects.
        """

        if not documents:
            logger.warning("No documents received for ingestion.")
            return

        logger.info("Starting ingestion pipeline.")

        try:
            # Metadata Enrichment
            enriched_documents = self.metadata_extractor.extract(documents)


            # Chunking
            # chunked_documents = self._chunk_documents(enriched_documents)

            chunked_documents = self.chunking_pipeline.split_text("\n\n".join(doc.page_content for doc in enriched_documents))

            logger.info(
                "Generated %d chunks from %d documents.",
                len(chunked_documents),
                len(enriched_documents),
            )

            # Generate Embeddings
            texts = [doc.page_content for doc in chunked_documents]

            embeddings = self.embedding_pipeline.embed_documents(texts)

            # --------------------------------------------------
            # Step 4: Persist to Vector Store
            # --------------------------------------------------
            self.vector_store.add_documents(
                documents=chunked_documents,
                embeddings=embeddings,
            )

            logger.info(
                "Successfully ingested %d chunks.",
                len(chunked_documents),
            )

        except Exception:
            logger.exception("Document ingestion failed.")
            raise
