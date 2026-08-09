import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

from ingestion.base_loader import BaseLoader


logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    '''
        Class for loading PDF documents using PyMuPDFLoader and PyPDFLoader.
    '''

    def load(self, path: Path):

        file_type = path.suffix.lower()
        logger.info( "Loading file: %s | Type: %s", path.name, file_type )

        try:
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()

        except Exception:
            loader = PyPDFLoader(str(path))
            docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = path.name
            doc.metadata["file_path"] = str(path)
            doc.metadata["file_type"] = file_type

        return docs

