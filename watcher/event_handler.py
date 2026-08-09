import hashlib
import logging
from pathlib import Path
import threading
import time

from watchdog.events import FileSystemEventHandler

from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.document_loader import DocumentLoader
from ingestion.loader_registry import LoadRegistry


logger = logging.getLogger(__name__)

DEBOUNCE_DELAY = 2.0  # seconds to wait before processing a modified file


class CorporaEventHandler(FileSystemEventHandler):

    def __init__(self, pipeline: IngestionPipeline):

        self.pipeline = pipeline
        self.loader = DocumentLoader()

        self.file_hashes = {}  # to track file content changes

        self._debounce_timers: dict[str, threading.Timer] = {}  # to track debounce timers for modified files
        self._lock = threading.Lock()  # to synchronize access to debounce timers


    def on_created(self, event):
        if event.is_directory:
            return
            
        self._schedule(Path(event.src_path), deleted=False)

    def on_modified(self, event):
        if event.is_directory:
            return
            
        self._schedule(Path(event.src_path), deleted=False)
        
    def on_deleted(self, event):
        if event.is_directory:
            return
            
        self._schedule(Path(event.src_path), deleted=True)

        
    def _schedule(self, file_path: Path, deleted: bool):
        """Cancel any pending timer for this file and start a fresh one."""

        key = str(file_path)

        with self._lock:
            existing = self._debounce_timers.get(key)

            if existing:
                existing.cancel()

            timer = threading.Timer(DEBOUNCE_DELAY, self._process_event, kwargs={"file_path": file_path, "deleted": deleted})

            self._debounce_timers[key] = timer
            timer.start()


    def _process_event(self, file_path: Path, deleted: bool):
        '''Cancel once, after debounce window closes'''

        key = str(file_path)

        with self._lock:
            self._debounce_timers.pop(key, None)

        if deleted:
            self._handle_deleted(file_path)

        else:
                self._handle_created_or_modified(file_path)


    def _handle_created_or_modified(self, file_path: Path):

        if not file_path.exists():
            return
            
        new_file_hash = self._compute_file_hash(file_path)
        old_file_hash = self.file_hashes.get(file_path.name)

        if old_file_hash == new_file_hash:
            logger.info(f"No real change detected (content unchanged): {file_path}")
            return
            
        if old_file_hash is not None:
            logger.info(f"Document modified, replacing embeddings: {file_path}")

            # delete old embeddings
            self.pipeline.vector_store.delete_by_source(file_path.name)

        else:
            logger.info(f"New document detected: {file_path}")

        documents = self.loader.load_file(file_path)
        self.pipeline.process_documents(documents)
        self.file_hashes[file_path.name] = new_file_hash


    def _handle_deleted(self, file_path: Path):
        logger.info(f"Document deleted: {file_path}")
        self.pipeline.vector_store.delete_by_source(file_path.name)
        self.file_hashes.pop(file_path.name, None)


    '''def on_created(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not file_path.exists():
            return
        
        file_hash = self._compute_file_hash(file_path)

        if file_path.name not in self.file_hashes or self.file_hashes[file_path.name] != file_hash:
            logger.info(f"New document detected: {file_path}")
            documents = self.loader.load_single_document(file_path)
            self.pipeline.process_documents(documents)

            self.file_hashes[file_path.name] = file_hash

        else:
            # Already known file (rare case)
            logger.info(f"Create event ignored (already known): {file_path}")


    def on_modified(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if not file_path.exists():
            return
        
        new_file_hash = self._compute_file_hash(file_path)
        old_hash_file = self.file_hashes.get(file_path.name)

        # Only process if content has changed
        if old_hash_file is None:

            # File was not tracked before, treat as new
            logger.info(f"Modified document (new file) detected: {file_path}")

            documents = self.loader.load_single_document(file_path)
            self.pipeline.process_documents(documents)

            self.file_hashes[file_path.name] = new_file_hash

        elif old_hash_file != new_file_hash:

            logger.info(f"Document modified: {file_path}")

            # delete old embeddings
            self.pipeline.vector_store.delete_by_source(file_path.name)

            # reprocess document
            documents = self.loader.load_single_document(file_path)
            self.pipeline.process_documents(documents)

            self.file_hashes[file_path.name] = new_file_hash
        
        else:
            logger.info(f"No real change detected (content unchanged): {file_path}")
        

    def on_deleted(self, event):

        if event.is_directory:
            return

        file_path = Path(event.src_path)

        logger.info(f"Document deleted: {file_path}")

        self.pipeline.vector_store.delete_by_source(file_path.name)'''


    def _compute_file_hash(self, file_path: Path) -> str:

        hasher = hashlib.md5()

        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()
