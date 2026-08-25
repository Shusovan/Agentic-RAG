import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.answer_generation_agent import AnswerGenerationAgent
from agents.routing_agent import RoutingAgent
from agents.retriever_agent import RetrievalAgent
from graph.state import RAGState
from agents.query_understanding_agent import QueryUnderstandingAgent
from llm.generative_llm import GenerativeLLM
from llm.query_understanding_llm import QueryUnderstandingLLM
from rag import vectorstore
from rag.retriever import Retriever
from schemas.rag_schema import Route
from tools.BM25_retriever_tool import BM25RetrieverTool
from tools.hybrid_retriever_tool import HybridRetrieverTool
from tools.vector_retriever_tool import VectorRetrieverTool
from config.vector_dependency import embedding_pipeline, vector_store, bm25_store


load_dotenv()


class Flow:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")

        # Query Understanding
        self.query_llm = QueryUnderstandingLLM(api_key=api_key)
        self.query_agent = QueryUnderstandingAgent(self.query_llm)

        # Routing Agent
        self.routing_agent = RoutingAgent()

        # Retrieval
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedding_pipeline = embedding_pipeline

        self.retriever = Retriever(vector_store, embedding_pipeline)
        self.vector_retriever_tool = VectorRetrieverTool(retriever=self.retriever)
        self.bm25_retriever_tool = BM25RetrieverTool(bm25_store=self.bm25_store)
        self.hybrid_retriever_tool = HybridRetrieverTool(vector_tool=self.vector_retriever_tool,
                                                bm25_retriever=self.bm25_retriever_tool)

        # Retrieval Agent
        self.retrieval_agent = RetrievalAgent(
            vector_retriever_tool=self.vector_retriever_tool,
            bm25_retriever_tool=self.bm25_retriever_tool,
            hybrid_retriever_tool = self.hybrid_retriever_tool,
            top_k=5,
            score_threshold=0.0,
            minimum_retrieval_count=1,
            quality_threshold=0.45,
            max_attempts=3)

        # Generation LLM
        self.generative_llm = GenerativeLLM(api_key=api_key)
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
    
    
    def route_decision(self, state: RAGState) -> str:

        if state.route == Route.RAG:
            return "rag"

        if state.route == Route.LLM:
            return "llm"

        if state.route == Route.CLARIFY:
            return "clarify"

        return "fallback"
    

    # Retrieval Node
    def retrieval_node(self, state: RAGState):

        if state.structured_query is None:
            raise ValueError("structured_query is None")

        documents = self.retrieval_agent.retrieve(state.structured_query)

        state.retrieved_documents = documents

        state.confidence = (max(doc.similarity_score for doc in documents)
            if documents else 0.0)

        return state


    # RAG Generation Node
    def rag_generation_node(self, state: RAGState) -> RAGState:

        if state.structured_query is None:
            raise ValueError("structured_query is None")

        state.final_answer = (self.answer_generation_agent.generate_rag_answer(
                structured_query=state.structured_query,
                documents=state.retrieved_documents,
            )
        )

        return state


    # Generation Node
    def llm_generation_node(self, state: RAGState) -> RAGState:

        state.final_answer = (
            self.answer_generation_agent.generate_general_answer(query=state.query)
        )

        return state


    # Clarification Node
    def clarify_node(self, state: RAGState) -> RAGState:

        state.final_answer = ("Could you clarify your question?")

        return state


    # Fallback Node
    def fallback_node(self, state: RAGState) -> RAGState:

        state.final_answer = ("This request is outside the supported scope.")

        return state  
    

    def _build_graph(self):

        builder = StateGraph(RAGState)

        builder.add_node("query_understanding", self.query_understanding_node)

        builder.add_node("routing", self.routing_node)

        builder.add_node("retrieval", self.retrieval_node)

        builder.add_node("rag_generation", self.rag_generation_node)

        builder.add_node("llm_generation", self.llm_generation_node)

        builder.add_node("clarify", self.clarify_node)

        builder.add_node("fallback", self.fallback_node)

        builder.set_entry_point("query_understanding")

        builder.add_edge("query_understanding", "routing")

        builder.add_conditional_edges(
            "routing",
            self.route_decision,
            {
                "rag": "retrieval",
                "llm": "llm_generation",
                "clarify": "clarify",
                "fallback": "fallback",
            }
        )

        builder.add_edge("retrieval", "rag_generation")

        builder.add_edge("rag_generation", END)

        builder.add_edge("llm_generation", END)

        builder.add_edge("clarify", END)

        builder.add_edge("fallback", END)


        return builder.compile()

    
    def run(self, query: str) -> RAGState:

        initial_state = RAGState(query=query)

        result = self.graph.invoke(initial_state)

        return RAGState(**result)