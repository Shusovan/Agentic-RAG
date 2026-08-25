import json
import logging
import os
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)


class BM25Store:
    """
        Maintains a BM25 index over the same document chunks
        that are stored in the vector store.

        BM25 is lexical retrieval:
            query terms -> matching document chunks

        Vector search is semantic retrieval:
            query embedding -> semantically similar chunks
    """

    def __init__(self, persist_path: str = "./qdrant_db/bm25_index.json"):

        self.persist_path = persist_path
        self.documents: List[Dict[str, Any]] = []
        self.bm25 = None
        self._load()


    # Tokenization
    def _tokenize(self, text: str) -> List[str]:
        
        # find words and convert to lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())

        return tokens


    # add documents
    def add_documents(self, documents: List[Any]) -> None:

        for docs in documents:

            document_id = getattr(docs, "id", None)

            if document_id is None:
                document_id = docs.metadata.get("chunk_id", None)

            if document_id is None:
                logger.warning("Document missing 'id' and 'chunk_id'. Skipping.")
                raise ValueError("Each document must contain a unique chunk_id")

            self.documents.append({
                "id": document_id,
                "content": docs.page_content,
                "metadata": dict(docs.metadata or {}),
            })

        self._rebuild()
        self._persist()

        logger.info("Added %d documents to BM25 index",len(documents))


    # delete documents from source
    def delete_by_source(self, source: str) -> None:

        before = len(self.documents)

        self.documents = [doc for doc in self.documents 
                            if doc["metadata"].get("source") != source]

        after = len(self.documents)

        deleted = before - after

        self._rebuild()
        self._persist()

        logger.info("Deleted %d documents from BM25 index for source: %s", deleted, source)


    # search documents
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        if not self.documents or self.bm25 is None:
            logger.warning("BM25 index is empty. No results to return.")
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked_index = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []

        rank = 1

        for index in ranked_index:

            bm25_score = float(scores[index])

            if bm25_score <= 0:
                continue

            doc = self.documents[index]

            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "bm25_score": bm25_score,
                "rank": rank
            })

            rank += 1

        # for rank, index in enumerate(ranked_index, start=1):

        #     bm25_score = float(scores[index])

        #     doc = self.documents[index]
            
        #     results.append({
        #         "id": doc["id"],
        #         "content": doc["content"],
        #         "metadata": doc["metadata"],
        #         "bm25_score": bm25_score,
        #         "rank": rank
        #     })

        return results


    def _rebuild(self) -> None:

        if not self.documents:
            logger.warning("BM25 index is empty.")
            self.bm25 = None
            return

        tokenized_documents = [
            self._tokenize(doc["content"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)


    # Persistence
    def _persist(self) -> None:

        directory = os.path.dirname(self.persist_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(self.persist_path, "w", encoding="utf-8") as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=4)


    def _load(self) -> None:

        if not os.path.exists(self.persist_path):
            logger.warning("Persisted file not found.")
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as file:
                self.documents = json.load(file)

            self._rebuild()

            logger.info("Loaded %d documents into BM25 index", len(self.documents),)

        except Exception:
            logger.exception("Failed to load BM25 index")

            self.documents = []
            self.bm25 = None 
