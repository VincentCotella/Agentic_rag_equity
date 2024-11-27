# agents/report_part2_agent.py

from .base_agent import BaseAgent
import json

class ReportPart2Agent(BaseAgent):
    def __init__(self, llm, structured_agent, unstructured_agent, company_name: str, status_placeholder=None):
        """
        Initialize the agent for generating Part II of the report (Key Financial Figures).

        Args:
            llm: Instance of the language model used by the agent.
            structured_agent: Instance of the StructuredDataAgent.
            unstructured_agent: Instance of the UnstructuredDataAgent.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        super().__init__(name="ReportPart2Agent", llm=llm)
        self.goal = "Generate Part II of the report: Key Financial Figures."
        self.backstory = "You are a financial analyst tasked with extracting key financial figures."
        self.structured_agent = structured_agent
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

    def generate_response(self) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart2Agent: Starting Part II generation...")

        # **Subtasks for Part II**
        financial_fields = {
            "Revenue": f"Retrieve the most recent total revenue of {self.company_name} as stated in the report.",
            "Net Income": f"Retrieve the most recent net income or net loss of {self.company_name} as stated in the report.",
            "Cash": f"Retrieve the amount of available cash and cash equivalents for {self.company_name}.",
            "Debt": f"Retrieve the total amount of debt for {self.company_name}.",
            "Equity": f"Retrieve the total equity amount for {self.company_name}.",
            "Operating Cash Flow": f"Retrieve the cash flow from operating activities for {self.company_name}."
        }

        results = {}

        # Process each financial field
        for field, query in financial_fields.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart2Agent: Processing {field}...")

            # Retrieve unstructured data
            unstructured_chunks = self.unstructured_agent.generate_response(query)
            unstructured_data = " ".join(unstructured_chunks)

            # Retrieve structured data
            structured_data_df = self.structured_agent.generate_response(self.company_name)
            structured_data_json = structured_data_df.to_json(orient='records')

            # Combine data and ask LLM to extract the field
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Extract the {field} of {self.company_name}.\n\n"
                f"Structured Data:\n{structured_data_json}\n\n"
                f"Unstructured Data:\n{unstructured_data}\n\n"
                f"Please provide the {field} based on the data above, including the units (e.g., USD).\n"
                f"If not available, respond with 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[field] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart2Agent: Part II generation completed.")

        return results
