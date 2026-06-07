import logging
from pathlib import Path
import time

from watchdog.observers import Observer
from watcher.event_handler import CorporaEventHandler
from ingestion.ingestion_pipeline import IngestionPipeline
from config.vector_dependency import vector_store, embedding_pipeline


logger = logging.getLogger(__name__)


class CorporaWatcher:

    def __init__(self, folder_path: str):

        self.folder_path = Path(folder_path)
        self.folder_path.mkdir(parents=True, exist_ok=True)

        pipeline = IngestionPipeline(vector_store=vector_store, embedding_pipeline=embedding_pipeline)

        self.event_handler = CorporaEventHandler(pipeline)

        self.observer = Observer()


    def start(self):

        logger.info(f"Starting watcher for folder: {self.folder_path}")

        self.observer.schedule(self.event_handler, str(self.folder_path), recursive=False)

        self.observer.start()

        try:

            while True:
                time.sleep(1)

        except KeyboardInterrupt:

            self.observer.stop()

        self.observer.join()
