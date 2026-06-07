from abc import ABC, abstractmethod


class RetrieverTool(ABC):

    @abstractmethod
    def retrieve(self, query: str):
        '''
            - Take in a query
            - Generate embedding
        '''
        pass