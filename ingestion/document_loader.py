from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, Docx2txtLoader


class DocumentLoader:

    def load_documents_from_directory(self, directory_path: Path):

        documents = []

        documents.extend(self.parse_pdf(directory_path))
        documents.extend(self.parse_docx(directory_path))
        documents.extend(self.parse_txt(directory_path))

        return documents
    

    def load_single_document(self, directory_path: Path):

        suffix = directory_path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(directory_path)

        elif suffix == ".docx":
            return self._load_docx(directory_path)

        elif suffix == ".txt":
            return self._load_txt(directory_path)

        else:
            return []


    def _load_pdf(self, directory_path: Path):

        try:
            loader = PyMuPDFLoader(str(directory_path))
            docs = loader.load()

        except Exception:
            loader = PyPDFLoader(str(directory_path))
            docs = loader.load()

        for d in docs:
            d.metadata["source"] = directory_path.name

        return docs


    def _load_docx(self, directory_path: Path):

        loader = Docx2txtLoader(str(directory_path))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = directory_path.name

        return docs


    def _load_txt(self, directory_path: Path):

        loader = TextLoader(str(directory_path))
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = directory_path.name

        return docs