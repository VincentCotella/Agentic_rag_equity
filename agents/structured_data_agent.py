# agents/structured_data_agent.py

from .base_agent import BaseAgent
import pandas as pd

class StructuredDataAgent(BaseAgent):
    def __init__(self, structured_data: pd.DataFrame):
        """
        Initialize the agent for structured data retrieval.

        Args:
            structured_data (pd.DataFrame): The structured data as a DataFrame.
        """
        super().__init__(name="StructuredDataAgent", llm=None)
        self.goal = "Retrieve relevant structured financial data."
        self.backstory = "You are an expert in financial data retrieval, specializing in providing relevant financial data upon request."
        self.structured_data = structured_data

    def retrieve_relevant_data(self, query: str) -> pd.DataFrame:
        """
        Retrieve relevant structured data based on the query.

        Args:
            query (str): The query to search for relevant data.

        Returns:
            pd.DataFrame: Relevant structured data.
        """
        filtered_data = self.structured_data[
            self.structured_data['Name'].str.contains(query, case=False, na=False) |
            self.structured_data['Symbol'].str.contains(query, case=False, na=False)
        ]
        return filtered_data

    def generate_response(self, query: str) -> pd.DataFrame:
        """
        Retrieves relevant structured data based on the query.

        Args:
            query (str): The query to retrieve relevant information.

        Returns:
            pd.DataFrame: Relevant structured data.
        """
        relevant_data = self.retrieve_relevant_data(query)
        return relevant_data
