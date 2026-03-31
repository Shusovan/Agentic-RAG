import shutil
import logging
from pathlib import Path
from fastapi import UploadFile

from config.settings import settings


logger = logging.getLogger(__name__)


class FileService:

    def __init__(self):

        self.corpora_path = settings.CORPORA_PATH

        self.corpora_path.mkdir(parents=True, exist_ok=True)


    def save_file(self, file: UploadFile) -> str:
        """
        Save uploaded file to Corpora directory
        """

        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {file_extension}"
            )

        file_path = self.corpora_path / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved to {file_path}")

        return str(file_path)
