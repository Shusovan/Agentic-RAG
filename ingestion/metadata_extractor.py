import logging
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Enriches document metadata before chunking.
    """

    def extract(self, documents: List[Document]) -> List[Document]:

        for document in documents:

            metadata = dict(document.metadata)

            metadata.setdefault("source", "unknown")

            metadata.setdefault("source_type", "unknown")

            metadata["ingestion_time"] = datetime.now(
                timezone.utc
            ).isoformat()

            metadata["content_length"] = len(document.page_content)

            metadata["word_count"] = len(
                document.page_content.split()
            )

            document.metadata = metadata

        logger.info(
            "Metadata extracted for %d documents",
            len(documents),
        )

        return documents
