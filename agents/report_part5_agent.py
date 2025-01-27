# agents/report_part5_agent.py

import logging
from .base_agent import BaseAgent
from core.edgar_direct_manager import EdgarDirectManager

logger = logging.getLogger(__name__)

class ReportPart5Agent(BaseAgent):
    def __init__(self, llm, unstructured_agent, company_name: str, status_placeholder=None):
        super().__init__(name="ReportPart5Agent", llm=llm)
        self.goal = "Generate Part V of the report: Risks and Challenges."
        self.backstory = "You are a financial analyst tasked with identifying the risks and challenges."
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
            self.status_placeholder.info("ReportPart5Agent: Starting Part V generation...")

        subtasks = {
            "Regulatory Risks": f"Identify any regulatory risks that {self.company_name} is facing.",
            "Market Risks": f"Describe the market risks affecting {self.company_name}.",
            "Operational Challenges": f"Outline operational challenges mentioned in the 10-K.",
            "Financial Risks": f"Highlight any financial risks disclosed by {self.company_name}."
        }

        # On suppose la plupart des risques sont dans "Item 1A"
        # Si vous jugez que d'autres items contiennent des risk factors, ajoutez-les
        item_mapping = {
            "Regulatory Risks": ["Item 1A"],
            "Market Risks": ["Item 1A"],
            "Operational Challenges": ["Item 1A"],
            "Financial Risks": ["Item 1A"]
        }

        results = {}
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart5Agent: Processing {section}...")

            if use_rag:
                chunks = self.unstructured_agent.generate_response(query)
                context_data = " ".join(chunks)
            else:
                items_to_join = item_mapping.get(section, [])
                context_data = self.edgar_manager.get_items_concat(items_to_join)

            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Provide insights on {section} for {self.company_name}.\n\n"
                f"Context:\n{context_data}\n\n"
                f"Please write a detailed analysis of {section.lower()} based on the data above.\n"
                f"If information is not available, indicate 'Not Available'."
            )
            response = self.llm(prompt).strip()
            results[section] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart5Agent: Part V generation completed.")

        return results
