from pathlib import Path


class Settings:

    CORPORA_PATH = Path("Corpora")

    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    MAX_FILE_SIZE_MB = 50


settings = Settings()