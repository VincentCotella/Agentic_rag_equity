# agents/report_part3_agent.py

from .base_agent import BaseAgent
import json

class ReportPart3Agent(BaseAgent):
    def __init__(self, llm, structured_agent, unstructured_agent, company_name: str, status_placeholder=None):
        """
        Initialize the agent for generating Part III of the report (Annual Performance Analysis).

        Args:
            llm: Instance of the language model used by the agent.
            structured_agent: Instance of the StructuredDataAgent.
            unstructured_agent: Instance of the UnstructuredDataAgent.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        super().__init__(name="ReportPart3Agent", llm=llm)
        self.goal = "Generate Part III of the report: Annual Performance Analysis."
        self.backstory = "You are a financial analyst tasked with analyzing the company's annual performance."
        self.structured_agent = structured_agent
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

    def generate_response(self) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart3Agent: Starting Part III generation...")

        # **Subtasks for Part III**
        subtasks = {
            "Performance Summary": f"Summarize the annual performance of {self.company_name}. Include key metrics and overall performance highlights.",
            "Performance Drivers": f"Identify the main factors that contributed to {self.company_name}'s performance during the year.",
            "Outlook": f"Describe the company's outlook or future projections as stated in the report."
        }

        results = {}

        # Process each subtask
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart3Agent: Processing {section}...")

            # Retrieve unstructured data
            unstructured_chunks = self.unstructured_agent.generate_response(query)
            unstructured_data = " ".join(unstructured_chunks)

            # Retrieve structured data if applicable
            structured_data_df = self.structured_agent.generate_response(self.company_name)
            structured_data_json = structured_data_df.to_json(orient='records')

            # Combine data and ask LLM to generate the section
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Provide the {section} for {self.company_name}.\n\n"
                f"Structured Data:\n{structured_data_json}\n\n"
                f"Unstructured Data:\n{unstructured_data}\n\n"
                f"Please write a detailed {section.lower()} based on the data above.\n"
                f"If information is not available, indicate 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[section] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart3Agent: Part III generation completed.")

        return results
