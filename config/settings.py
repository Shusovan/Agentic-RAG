from pathlib import Path


class Settings:

    # Corpora settings
    CORPORA_PATH = Path("Corpora")
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
    MAX_FILE_SIZE_MB = 50

    # Model settings
    MODEL: str = "qwen/qwen3-32b"
    TEMPERATURE: float = 0.0
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.3
    MAX_RETRIES: int = 3
    MAX_ITERATIONS: int = 10

    # Intent routing constants
    RAG_INTENTS: frozenset = frozenset({"INTERNAL_QUESTION", "DOCUMENT_QUERY"})
    LLM_INTENTS: frozenset = frozenset({"GENERAL_QUESTION", "GREETING", "CHITCHAT"})
    CLARIFY_INTENTS: frozenset = frozenset({"AMBIGUOUS"})
    FALLBACK_INTENTS: frozenset = frozenset({"OUT_OF_SCOPE"})

    # Source routing constants
    RAG_SOURCES: frozenset = frozenset({"INTERNAL", "UNKNOWN"})
    LLM_SOURCES: frozenset = frozenset({"GENERAL"})


settings = Settings()