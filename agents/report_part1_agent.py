# agents/report_part1_agent.py

from .base_agent import BaseAgent
import json

class ReportPart1Agent(BaseAgent):
    def __init__(self, llm, structured_agent, unstructured_agent, company_name: str, status_placeholder=None):
        """
        Initialize the agent for generating Part I of the report (General Presentation).

        Args:
            llm: Instance of the language model used by the agent.
            structured_agent: Instance of the StructuredDataAgent.
            unstructured_agent: Instance of the UnstructuredDataAgent.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        super().__init__(name="ReportPart1Agent", llm=llm)
        self.goal = "Generate Part I of the report: General Presentation."
        self.backstory = "You are a financial analyst tasked with compiling general company information."
        self.structured_agent = structured_agent
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

    def generate_response(self) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart1Agent: Starting Part I generation...")

        # **Subtasks for Part I**
        subtasks = {
            "Ticker": f"Retrieve the ticker symbol of {self.company_name}.",
            "Name": f"Confirm the full company name for {self.company_name}.",
            "Country": f"Identify the country of domicile for {self.company_name}.",
            "Sector": f"Determine the business sector or industry of {self.company_name}.",
            "Description": f"Provide a concise description of {self.company_name} from the 10-K report.",
            "CEO": f"Find the name of the CEO of {self.company_name}.",
            "Headquarters": f"Locate the headquarters of {self.company_name}.",
            "Employees": f"Retrieve the number of employees of {self.company_name}."
        }

        # Initialize a dictionary to store results
        results = {}

        # Process each subtask
        for field, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart1Agent: Processing {field}...")

            # Retrieve data from unstructured data agent
            unstructured_chunks = self.unstructured_agent.generate_response(query)
            unstructured_data = " ".join(unstructured_chunks)

            # Retrieve data from structured data agent
            structured_data_df = self.structured_agent.generate_response(self.company_name)
            structured_data_json = structured_data_df.to_json(orient='records')

            # Combine data and ask LLM to extract the field
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Extract the {field} of {self.company_name}.\n\n"
                f"Structured Data:\n{structured_data_json}\n\n"
                f"Unstructured Data:\n{unstructured_data}\n\n"
                f"Please provide the {field} based on the data above.\n"
                f"If not available, respond with 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[field] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart1Agent: Part I generation completed.")

        return results
