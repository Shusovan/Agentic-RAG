import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.answer_generation_agent import AnswerGenerationAgent
from agents.routing_agent import RoutingAgent
from agents.vector_retriever_agent import VectorRetrieverAgent
from graph.state import RAGState
from agents.query_understanding_agent import QueryUnderstandingAgent
from llm.generative_llm import GenerativeLLM
from llm.query_understanding_llm import QueryUnderstandingLLM
from rag import vectorstore
from rag.retriever import Retriever
from schemas.rag_schema import Route
from tools.vector_retriever_tool import VectorRetrieverTool
from config.vector_dependency import embedding_pipeline, vector_store


load_dotenv()


class Flow:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")

        # Query Understanding LLM
        self.query_llm = QueryUnderstandingLLM(api_key=api_key)

        # Query Understanding Agent
        self.query_agent = QueryUnderstandingAgent(self.query_llm)

        # Routing Agent
        self.routing_agent = RoutingAgent()

        # Retrieval
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline

        self.retriever = Retriever(vector_store, embedding_pipeline)

        self.vector_retriever_tool = VectorRetrieverTool(retriever=self.retriever)

        self.vector_retriever_agent = VectorRetrieverAgent(
            retriever_tool=self.vector_retriever_tool
        )
        # self.vector_retriever_agent = VectorRetrieverAgent(retriever_tool=vector_retriever_tool)

        # Generation LLM
        self.generative_llm = GenerativeLLM(api_key=api_key)

        # Answer Generation
        self.answer_generation_agent = AnswerGenerationAgent(llm=self.generative_llm)

        self.graph = self._build_graph()


    def query_understanding_node(self, state: RAGState) -> RAGState:

        structured = self.query_agent.process_query(state.query)

        state.structured_query = structured

        return state
    

    def routing_node(self, state: RAGState) -> RAGState:

        structured = state.structured_query

        if structured is None:
            raise ValueError("structured_query missing")

        state.route = self.routing_agent.route(structured)

        return state
    
    
    def route_decision(
        self,
        state: RAGState
    ) -> str:

        if state.route == Route.RAG:
            return "rag"

        if state.route == Route.LLM:
            return "llm"

        if state.route == Route.CLARIFY:
            return "clarify"

        return "fallback"
    

    def vector_retrieval_node(self, state: RAGState):

        docs = self.vector_retriever_agent.retrieve(
            state.structured_query
        )

        state.retrieved_documents = docs

        return state
    

    def rag_generation_node(self, state: RAGState) -> RAGState:

        state.final_answer = (self.answer_generation_agent.generate_rag_answer(
                structured_query=state.structured_query,
                documents=state.retrieved_documents,
            )
        )

        return state


    def llm_generation_node(self, state: RAGState) -> RAGState:

        state.final_answer = (
            self.answer_generation_agent.generate_general_answer(
                query=state.query
            )
        )

        return state


    def clarify_node(
        self,
        state: RAGState
    ) -> RAGState:

        state.final_answer = (
            "Could you clarify your question?"
        )

        return state


    def fallback_node(self, state: RAGState) -> RAGState:

        state.final_answer = ("This request is outside the supported scope.")

        return state  
    

    def _build_graph(self):

        builder = StateGraph(RAGState)

        builder.add_node(
            "query_understanding",
            self.query_understanding_node
        )

        builder.add_node(
            "routing",
            self.routing_node
        )

        builder.add_node(
            "vector_retrieval",
            self.vector_retrieval_node
        )

        builder.add_node(
            "rag_generation",
            self.rag_generation_node
        )

        builder.add_node(
            "llm_generation",
            self.llm_generation_node
        )

        builder.add_node("clarify", self.clarify_node)

        builder.add_node("fallback", self.fallback_node)

        builder.set_entry_point(
            "query_understanding"
        )

        builder.add_edge(
            "query_understanding",
            "routing"
        )

        builder.add_conditional_edges(
            "routing",
            self.route_decision,
            {
                "rag": "vector_retrieval",
                "llm": "llm_generation",
                "clarify": "clarify",
                "fallback": "fallback",
            }
        )

        builder.add_edge("vector_retrieval", "rag_generation")

        builder.add_edge("rag_generation", END)

        builder.add_edge("llm_generation", END)

        builder.add_edge("clarify", END)

        builder.add_edge("fallback", END)


        return builder.compile()

    
    def run(self, query: str) -> RAGState:

        initial_state = RAGState(query=query)

        result = self.graph.invoke(initial_state)

        return RAGState(**result)