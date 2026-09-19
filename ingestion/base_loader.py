from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseLoader(ABC):
    """
        Base class for document loaders.
    """

    @abstractmethod
    def load(self, source) -> list[Document]:
        pass
