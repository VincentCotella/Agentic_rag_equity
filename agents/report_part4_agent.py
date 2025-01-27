# agents/report_part4_agent.py

import logging
from .base_agent import BaseAgent
from core.edgar_direct_manager import EdgarDirectManager

logger = logging.getLogger(__name__)

class ReportPart4Agent(BaseAgent):
    def __init__(self, llm, unstructured_agent, company_name: str, status_placeholder=None):
        super().__init__(name="ReportPart4Agent", llm=llm)
        self.goal = "Generate Part IV of the report: Market Position and Competitors."
        self.backstory = "You are a financial analyst tasked with analyzing the company's market position and competitors."
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

        self.edgar_manager = EdgarDirectManager(
            ticker_symbol=self.company_name,
            user_agent="vincent.cotella@edu.devinci.fr",
            report_type="10-K"
        )

    def generate_response(self, use_rag: bool, context_dict: dict) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart4Agent: Starting Part IV generation...")

        subtasks = {
            "Market Position": f"Describe {self.company_name}'s position in the market.",
            "Key Competitors": f"Identify the main competitors of {self.company_name}.",
            "Competitive Advantages": f"Explain the competitive advantages that {self.company_name} holds."
        }

        # PART IV => Item 15
        # Suppose que Market/competitors se trouve dans Item 7
        item_mapping = {
            "Market Position": ["Item 7"],
            "Key Competitors": ["Item 7"],
            "Competitive Advantages": ["Item 7"]
        }

        results = {}
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart4Agent: Processing {section}...")

            if use_rag:
                chunks = self.unstructured_agent.generate_response(query)
                context_data = " ".join(chunks)
            else:
                items_to_join = item_mapping.get(section, [])
                context_data = self.edgar_manager.get_items_concat(items_to_join)
                
            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Provide the {section} for {self.company_name}.\n\n"
                f"Context:\n{context_data}\n\n"
                f"Please write a detailed {section.lower()} based on the data above.\n"
                f"If information is not available, indicate 'Not Available'."
            )
            response = self.llm(prompt).strip()
            results[section] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart4Agent: Part IV generation completed.")

        return results
