from pathlib import Path

from ingestion.docx_loader import DOCXLoader
from ingestion.pdf_loader import PDFLoader
from ingestion.text_loader import TxtLoader


class LoadRegistry:
    """
        Registry for document loaders
    """

    def __init__(self):
        self.loaders = {
            "txt": TxtLoader(),
            "docx": DOCXLoader(),
            "pdf": PDFLoader()
        }


    def get_file_loader(self, path: Path):
        return self.loaders.get(path.suffix.lower().replace(".", ""), None)


    def get_web_loader(self):
        return self.web_loader