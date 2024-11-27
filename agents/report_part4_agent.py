# agents/report_part4_agent.py

from .base_agent import BaseAgent
import json

class ReportPart4Agent(BaseAgent):
    def __init__(self, llm, structured_agent, unstructured_agent, company_name: str, status_placeholder=None):
        """
        Initialize the agent for generating Part IV of the report (Market Position and Competitors).

        Args:
            llm: Instance of the language model used by the agent.
            structured_agent: Instance of the StructuredDataAgent.
            unstructured_agent: Instance of the UnstructuredDataAgent.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        super().__init__(name="ReportPart4Agent", llm=llm)
        self.goal = "Generate Part IV of the report: Market Position and Competitors."
        self.backstory = "You are a financial analyst tasked with analyzing the company's market position and its competitors."
        self.structured_agent = structured_agent
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

    def generate_response(self) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart4Agent: Starting Part IV generation...")

        # **Subtasks for Part IV**
        subtasks = {
            "Market Position": f"Describe {self.company_name}'s position in the market, including market share and strengths.",
            "Key Competitors": f"Identify the main competitors of {self.company_name} and their relative positions.",
            "Competitive Advantages": f"Explain the competitive advantages that {self.company_name} holds over its competitors."
        }

        results = {}

        # Process each subtask
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart4Agent: Processing {section}...")

            # Retrieve unstructured data
            unstructured_chunks = self.unstructured_agent.generate_response(query)
            unstructured_data = " ".join(unstructured_chunks)

            # No structured data used here, but if needed, you can retrieve it

            # Use LLM to generate the section
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Provide the {section} for {self.company_name}.\n\n"
                f"Unstructured Data:\n{unstructured_data}\n\n"
                f"Please write a detailed {section.lower()} based on the data above.\n"
                f"If information is not available, indicate 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[section] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart4Agent: Part IV generation completed.")

        return results
