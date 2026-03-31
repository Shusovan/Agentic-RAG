import logging
from datetime import datetime
from typing import List


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Extract and enrich metadata for documents
    """

    def extract(self, documents: List):

        enriched_docs = []

        for doc in documents:

            metadata = dict(doc.metadata)

            metadata["ingestion_time"] = datetime.utcnow().isoformat()

            metadata["content_length"] = len(doc.page_content)

            if "source" not in metadata:
                metadata["source"] = "unknown"

            doc.metadata = metadata

            enriched_docs.append(doc)

        logger.info(
            f"Metadata enriched for {len(enriched_docs)} documents"
        )

        return enriched_docs
