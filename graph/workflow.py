import os

from dotenv import load_dotenv

from graph.state import RAGState
from agents.query_understanding_agent import QueryUnderstandingAgent
from llm.query_understanding_llm import QueryUnderstandingLLM


load_dotenv()


class Flow:

    def __init__(self):
        
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        
        self.llm = QueryUnderstandingLLM(api_key=api_key)
        self.query_agent = QueryUnderstandingAgent(self.llm)

    
    def run(self, query: str):

        state = RAGState(query=query)

        structured_query = self.query_agent.process_query(query)

        state.structured_query = structured_query

        return state