import hashlib
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Optional

from watchdog.events import FileSystemEventHandler

from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.document_loader import DocumentLoader


logger = logging.getLogger(__name__)


DEBOUNCE_DELAY = 2.0


class CorporaEventHandler(FileSystemEventHandler):

    def __init__(self, pipeline: IngestionPipeline):

        self.pipeline = pipeline
        self.loader = DocumentLoader()

        # -------------------------------------------------
        # Track file content hashes
        # -------------------------------------------------

        self.file_hashes: dict[str, str] = {}

        # -------------------------------------------------
        # Debounce timers
        # -------------------------------------------------

        self._debounce_timers: dict[
            str,
            threading.Timer
        ] = {}

        self._lock = threading.Lock()

        # -------------------------------------------------
        # Ingestion queue
        #
        # Only ONE worker consumes this queue.
        # Therefore only one ingestion operation runs
        # at a time.
        # -------------------------------------------------

        self._ingestion_queue: queue.Queue[
            tuple[Path, bool]
        ] = queue.Queue()

        self._worker_thread = threading.Thread(
            target=self._ingestion_worker,
            name="corpora-ingestion-worker",
            daemon=True
        )

        self._worker_thread.start()

        logger.info("Corpora ingestion worker started")


    # WATCHDOG EVENTS
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


    # DEBOUNCE
    def _schedule(self, file_path: Path, deleted: bool):
        """
        Debounce filesystem events.

        Multiple events for the same file within the
        debounce window result in only one queued job.
        """

        key = str(file_path.resolve())

        with self._lock:
            existing_timer = (self._debounce_timers.get(key))

            if existing_timer:
                existing_timer.cancel()

            timer = threading.Timer(
                DEBOUNCE_DELAY,
                self._queue_event,
                kwargs={
                    "file_path": file_path,
                    "deleted": deleted
                }
            )

            self._debounce_timers[key] = timer

            timer.daemon = True
            timer.start()


    # QUEUE EVENT
    def _queue_event(self, file_path: Path, deleted: bool):
        """
        Called after the debounce period.

        IMPORTANT:
        This method does NOT perform ingestion.

        It only puts the job into the queue.
        """

        key = str(file_path.resolve())

        with self._lock:
            self._debounce_timers.pop(key, None)

        logger.info("Queueing filesystem event | file=%s | deleted=%s", file_path, deleted)

        self._ingestion_queue.put((file_path, deleted))


    # INGESTION WORKER
    def _ingestion_worker(self):
        """
        Single worker responsible for processing ingestion.

        This guarantees that only one ingestion operation
        interacts with Qdrant Local at a time.
        """

        while True:
            file_path, deleted = (self._ingestion_queue.get())

            try:
                self._process_event(file_path, deleted)

            except Exception:
                logger.exception("Unhandled error processing filesystem event: %s", file_path)

            finally:
                self._ingestion_queue.task_done()

  
    # PROCESS EVENT
    def _process_event(self, file_path: Path, deleted: bool):

        if deleted:
            self._handle_deleted(file_path)

        else:
            self._handle_created_or_modified(file_path)


    # CREATED / MODIFIED
    def _handle_created_or_modified(self, file_path: Path):

        if not file_path.exists():
            logger.warning("File no longer exists: %s", file_path)
            return

        try:
            new_file_hash = (self._compute_file_hash(file_path))

        except Exception:
            logger.exception("Failed to calculate file hash: %s", file_path)
            return

        # Use full path as the hash key
        key = str(file_path.resolve())

        old_file_hash = (self.file_hashes.get(key))

        # No actual content change
        if old_file_hash == new_file_hash:
            logger.info("No real change detected (content unchanged): %s", file_path)
            return

        # Existing document
        if old_file_hash is not None:
            logger.info("Document modified, replacing embeddings: %s", file_path)

            self.pipeline.vector_store.delete_by_source(file_path.name)

        # New document
        else:
            logger.info("New document detected: %s", file_path)

        # Load document
        try:
            documents = self.loader.load_file(file_path)

        except Exception:
            logger.exception("Failed to load document: %s", file_path)
            return

        if not documents:
            logger.warning("No documents generated from: %s", file_path)
            return

        # Ingest
        # This is now executed by ONE worker only.
        try:
            self.pipeline.process_documents(documents)

        except Exception:
            logger.exception("Document ingestion failed: %s", file_path)
            return

        # Update hash ONLY after successful ingestion
        self.file_hashes[key] = (new_file_hash)
        logger.info("Document successfully processed: %s", file_path)


    # deleted
    def _handle_deleted(self, file_path: Path):

        logger.info("Document deleted: %s",file_path)

        try:
            self.pipeline.vector_store.delete_by_source(file_path.name)

            key = str(file_path.resolve())

            self.file_hashes.pop(key, None)

            logger.info("Deleted document embeddings: %s", file_path)

        except Exception:
            logger.exception("Failed to delete embeddings: %s", file_path)


    # HASH
    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:

        hasher = hashlib.md5()

        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(4096),b""):
                hasher.update(chunk)

        return hasher.hexdigest()