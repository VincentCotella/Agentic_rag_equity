# agents/report_part3_agent.py

import logging
from .base_agent import BaseAgent
from core.edgar_direct_manager import EdgarDirectManager

logger = logging.getLogger(__name__)

class ReportPart3Agent(BaseAgent):
    def __init__(self, llm, unstructured_agent, company_name: str, status_placeholder=None):
        super().__init__(name="ReportPart3Agent", llm=llm)
        self.goal = "Generate Part III of the report: Annual Performance Analysis."
        self.backstory = "You are a financial analyst tasked with analyzing the company's annual performance."
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
            self.status_placeholder.info("ReportPart3Agent: Starting Part III generation...")

        subtasks = {
            "Performance Summary": f"Summarize the annual performance of {self.company_name}.",
            "Performance Drivers": f"Identify the main factors that contributed to {self.company_name}'s performance.",
            "Outlook": f"Describe the company's outlook or future projections from the 10-K."
        }

        # PART III => items 10..14, 
        # Mais souvent, la Perf. Analysis se trouve dans Item 7 
        # (Vous pouvez adapter selon votre 10-K)
        item_mapping = {
            "Performance Summary": ["Item 7", "Item 7A"],
            "Performance Drivers": ["Item 7"],
            "Outlook": ["Item 7"]
        }

        results = {}
        for section, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart3Agent: Processing {section}...")

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
            self.status_placeholder.success("ReportPart3Agent: Part III generation completed.")

        return results
