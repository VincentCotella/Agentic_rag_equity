# agents/unstructured_data_agent.py

from .base_agent import BaseAgent
from langchain.vectorstores import Chroma
from typing import List
from config import CHROMA_PATH, TOP_K
from embedding import get_embedding_function

class UnstructuredDataAgent(BaseAgent):
    def __init__(self):
        """
        Initialize the agent for unstructured data retrieval (RAG).
        """
        super().__init__(name="UnstructuredDataAgent", llm=None)
        self.goal = "Retrieve relevant unstructured financial document chunks."
        self.backstory = "You are an expert in financial document retrieval, specializing in extracting relevant information from 10-K filings."
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        self.top_k = TOP_K

    def retrieve_relevant_chunks(self, query: str) -> List[str]:
        """
        Retrieve relevant document chunks based on the query using similarity search in Chroma.
        """
        results = self.db.similarity_search(query, k=self.top_k)
        return [doc.page_content for doc in results]

    def generate_response(self, query: str) -> List[str]:
        """
        Retrieves relevant unstructured data based on the query.
        """
        relevant_chunks = self.retrieve_relevant_chunks(query)
        return relevant_chunks
