import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler

from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.document_loader import DocumentLoader


logger = logging.getLogger(__name__)


class CorporaEventHandler(FileSystemEventHandler):

    def __init__(self, pipeline: IngestionPipeline):

        self.pipeline = pipeline
        self.loader = DocumentLoader()


    def on_created(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        logger.info(f"New document detected: {file_path}")

        documents = self.loader.load_single_document(file_path)

        self.pipeline.process_documents(documents)


    def on_modified(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        logger.info(f"Document modified: {file_path}")

        # remove old embeddings
        self.pipeline.vector_store.delete_by_source(file_path.name)

        documents = self.loader.load_single_document(file_path)

        self.pipeline.process_documents(documents)


    def on_deleted(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        logger.info(f"Document deleted: {file_path}")

        self.pipeline.vector_store.delete_by_source(file_path.name)
