from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


'''class ChunkingPipeline:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


    def split_text(self, text: str):

        chunks = self.text_splitter.create_documents([text])

        return chunks'''

class ChunkingPipeline:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document],) -> List[Document]:

        if not documents:
            return []

        return self.text_splitter.split_documents(documents)