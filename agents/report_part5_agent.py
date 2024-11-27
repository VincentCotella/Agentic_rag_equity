# agents/report_part5_agent.py

from .base_agent import BaseAgent
import json

class ReportPart5Agent(BaseAgent):
    def __init__(self, llm, structured_agent, unstructured_agent, company_name: str, status_placeholder=None):
        """
        Initialize the agent for generating Part V of the report (Risks and Challenges).

        Args:
            llm: Instance of the language model used by the agent.
            structured_agent: Instance of the StructuredDataAgent.
            unstructured_agent: Instance of the UnstructuredDataAgent.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        super().__init__(name="ReportPart5Agent", llm=llm)
        self.goal = "Generate Part V of the report: Risks and Challenges."
        self.backstory = "You are a financial analyst tasked with identifying the risks and challenges faced by the company."
        self.structured_agent = structured_agent
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

    def generate_response(self) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart5Agent: Starting Part V generation...")

        # **Subtasks for Part V**
        subtasks = {
            "Regulatory Risks": f"Identify any regulatory risks that {self.company_name} is facing.",
            "Market Risks": f"Describe the market risks affecting {self.company_name}.",
            "Operational Challenges": f"Outline operational challenges mentioned in the report.",
            "Financial Risks": f"Highlight any financial risks disclosed by {self.company_name}."
        }

        results = {}

        # Process each subtask
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart5Agent: Processing {section}...")

            # Retrieve unstructured data
            unstructured_chunks = self.unstructured_agent.generate_response(query)
            unstructured_data = " ".join(unstructured_chunks)

            # Use LLM to generate the section
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Provide insights on {section} for {self.company_name}.\n\n"
                f"Unstructured Data:\n{unstructured_data}\n\n"
                f"Please write a detailed analysis of {section.lower()} based on the data above.\n"
                f"If information is not available, indicate 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[section] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart5Agent: Part V generation completed.")

        return results
