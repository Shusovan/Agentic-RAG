import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class EmbeddingPipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = None
        self._load_model()


    def _load_model(self):

        try:
            logger.info(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"Model loaded successfully. Dimension: {dim}")

        except Exception as e:
            raise ValueError(f"Failed to load sentence transformer model '{self.model_name}': {e}")


    def embed_documents(self, documents: List[str]) -> np.ndarray:

        if not self.model:
            raise ValueError("Embedding model not loaded")

        embeddings = self.model.encode(
            documents,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings