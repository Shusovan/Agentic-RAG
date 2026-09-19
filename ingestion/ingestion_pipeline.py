import hashlib
import json
import logging
from pathlib import Path
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
        2. Document ID assignment
        3. Chunking
        4. Embedding generation
        5. Vector store ingestion
    """

    def __init__(
        self,
        vector_store,
        embedding_pipeline,
        metadata_extractor: MetadataExtractor | None = None,
        chunking_pipeline: ChunkingPipeline | None = None,
        chunk_registry_path: str = "evaluation/datasets/corpus_chunks.json"):

        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

        self.metadata_extractor = (metadata_extractor or MetadataExtractor())

        self.chunking_pipeline = (chunking_pipeline or ChunkingPipeline())

        self.chunk_registry_path = Path(chunk_registry_path)
        self.chunk_registry_path.parent.mkdir(parents=True, exist_ok=True,)


    def _generate_document_id(self, source: str) -> str:
        """
            Generate a deterministic document ID from the source.
            The same source will always generate the same document_id.
        """

        if not source:
            logger.error("Document source cannot be empty")
            raise ValueError("Document source cannot be empty")

        normalized_source = source.strip().lower()

        return hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()[:16]


    def _assign_document_ids(self, documents: List[Document],) -> List[Document]:
        """
        Assign a stable document_id to every source document.
        """

        for document in documents:

            source = document.metadata.get("source")

            if not source:
                logger.error("Document source cannot be empty")
                raise ValueError("Metadata must contain source")

            document_id = self._generate_document_id(source=source)

            document.metadata["document_id"] = document_id

            logger.info(
                "Assigned document_id=%s to source=%s",
                document_id,
                source,)

        return documents


    def _chunk_documents(self, documents: List[Document],) -> List[Document]:
        """
        Chunk documents while preserving document metadata.

        Every generated chunk receives the document_id
        of its original source document.
        """

        chunked_docs = []

        chunk_counter = {}

        for document in documents:

            document_id = document.metadata.get("document_id")

            source = document.metadata.get("source")

            if not document_id:
                raise ValueError("Document metadata must contain document_id")

            if not source:
                raise ValueError("Document metadata must contain source")

            chunk_counter.setdefault( document_id, 0)

            chunks = self.chunking_pipeline.split_text(document.page_content)

            for chunk_index, chunk in enumerate( chunks, start=1,):
                chunk_counter[document_id] += 1 

                chunk_index = chunk_counter[document_id]

                if isinstance(chunk, Document):
                    chunk_document = chunk

                else:
                    chunk_document = Document(page_content=chunk)

                chunk_document.metadata.update(document.metadata)

                chunk_document.metadata["document_id"] = document_id

                chunk_document.metadata["chunk_id"] = (f"{document_id}_chunk_{chunk_index}")

                chunk_document.metadata["chunk_index"] = chunk_index

                logger.info("Created chunk | document_id=%s | chunk_id=%s | chunk_index=%s | source=%s",
                        document_id,
                        chunk_document.metadata["chunk_id"],
                        chunk_index,
                        source)

                chunk_document.metadata["source"] = source

                chunked_docs.append(chunk_document)

        return chunked_docs


    def _save_chunk_registry(self, chunked_documents: List[Document]) -> None:
        """
            Save chunk information for evaluation dataset generation.

            The registry contains the actual chunk content together
            with the IDs assigned during ingestion.

            This allows the evaluation dataset generator to know
            exactly which chunk_id belongs to which chunk.

            Existing registry entries are preserved.
        """

        registry = []

        # if registry already exists
        if self.chunk_registry_path.exists():
            try:
                with self.chunk_registry_path.open("r", encoding="utf-8") as file:
                    registry = json.load(file)

                if not isinstance(registry, list):
                    logger.warning("Existing chunk registry is not a list. "
                        "Starting with empty registry.")

                    registry = []

            except json.JSONDecodeError:

                logger.warning("Could not parse existing chunk registry. "
                    "Starting with empty registry.")

                registry = []

        # remove old entries of documents being re-ingested
        document_ids = {str(document.metadata["document_id"])
                                for document in chunked_documents}

        registry = [
            entry 
            for entry in registry 
            if str(entry.get("document_id"))
            not in document_ids
        ]

        # all generated chunks
        for document in chunked_documents:

            metadata = document.metadata

            registry.append(
                {
                    "document_id": metadata.get("document_id"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "source": metadata.get("source"),
                    "content": document.page_content,
                    "metadata": dict(metadata),
                }
            )

        # save the chunks
        with self.chunk_registry_path.open("w", encoding="utf-8",) as file:
            json.dump(registry, file, indent=4, ensure_ascii=False,)

        logger.info("Chunk registry saved | chunks=%d | path=%s",
            len(chunked_documents),
            self.chunk_registry_path,)
        

    def process_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        Execute the complete ingestion pipeline.

        Args:
            documents: List of LangChain Document objects.
        """

        if not documents:
            logger.warning(
                "No documents received for ingestion."
            )
            return

        logger.info(
            "Starting ingestion pipeline."
        )

        try:

            # -------------------------------------------------
            # 1. Metadata enrichment
            # -------------------------------------------------

            enriched_documents = (
                self.metadata_extractor.extract(
                    documents
                )
            )

            # -------------------------------------------------
            # 2. Assign stable document IDs
            # -------------------------------------------------

            enriched_documents = (
                self._assign_document_ids(
                    enriched_documents
                )
            )

            # 3. Chunk documents
            chunked_documents = (
                self._chunk_documents(
                    enriched_documents
                )
            )

            logger.info("Generated %d chunks from %d documents.",
                len(chunked_documents),
                len(enriched_documents),)

            if not chunked_documents:
                logger.error("No chunks generated")
                return

            self._save_chunk_registry(chunked_documents)

            # 4. Generate embeddings
            texts = [doc.page_content
                for doc in chunked_documents]

            embeddings = (self.embedding_pipeline.embed_documents(texts))

            # 5. Persist to vector store
            self.vector_store.add_documents(documents=chunked_documents, embeddings=embeddings,)

            logger.info("Successfully ingested %d chunks.",len(chunked_documents),)

        except Exception:
            logger.exception(
                "Document ingestion failed."
            )
            raise