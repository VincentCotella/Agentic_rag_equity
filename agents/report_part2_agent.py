# agents/report_part2_agent.py

import logging
from .base_agent import BaseAgent
from core.edgar_direct_manager import EdgarDirectManager

logger = logging.getLogger(__name__)

class ReportPart2Agent(BaseAgent):
    def __init__(self, llm, unstructured_agent, company_name: str, status_placeholder=None):
        super().__init__(name="ReportPart2Agent", llm=llm)
        self.goal = "Generate Part II of the report: Key Financial Figures."
        self.backstory = "You are a financial analyst tasked with extracting key financial figures."
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

        self.edgar_manager = EdgarDirectManager(
            ticker_symbol=self.company_name,
            user_agent="Vincent Cotella vincent.cotella@edu.devinci.fr",
            report_type="10-K"
        )

    def generate_response(self, use_rag: bool, context_dict: dict) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart2Agent: Starting Part II generation...")

        financial_fields = {
            "Revenue": f"Retrieve the most recent total revenue of {self.company_name}.",
            "Net Income": f"Retrieve the most recent net income of {self.company_name}.",
            "Cash": f"Retrieve the amount of available cash for {self.company_name}.",
            "Debt": f"Retrieve the total amount of debt for {self.company_name}.",
            "Equity": f"Retrieve the total equity amount for {self.company_name}.",
            "Operating Cash Flow": f"Retrieve the cash flow from operating activities for {self.company_name}."
        }

        # PART II => Items 5..9 => 
        # On associe chaque champ à quelques items (max 2-3)
        item_mapping = {
            "Revenue": ["Item 6", "Item 7"],
            "Net Income": ["Item 6", "Item 7"],
            "Cash": ["Item 7", "Item 7A"],
            "Debt": ["Item 7", "Item 7A"],
            "Equity": ["Item 7", "Item 7A"],
            "Operating Cash Flow": ["Item 7"]
        }

        results = {}
        for field, query in financial_fields.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart2Agent: Processing {field}...")

            if use_rag:
                chunks = self.unstructured_agent.generate_response(query)
                context_data = " ".join(chunks)
            else:
                items_to_join = item_mapping.get(field, [])
                context_data = self.edgar_manager.get_items_concat(items_to_join)

            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Extract the {field} of {self.company_name}.\n\n"
                f"Context:\n{context_data}\n\n"
                f"Please provide the {field} based on the data above, including units.\n"
                f"If not available, respond with 'Not Available'."
            )
            response = self.llm(prompt).strip()
            results[field] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart2Agent: Part II generation completed.")

        return results
