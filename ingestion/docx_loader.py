import logging
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader

from ingestion.base_loader import BaseLoader


logger = logging.getLogger(__name__)


class DOCXLoader(BaseLoader):

    def load(self, path: Path):

        file_type = path.suffix.lower()
        logger.info( "Loading file: %s | Type: %s", path.name, file_type )

        loader = Docx2txtLoader(str(path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path)
            doc.metadata["file_type"] = file_type

        return docs