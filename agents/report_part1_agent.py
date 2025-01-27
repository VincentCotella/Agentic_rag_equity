# agents/report_part1_agent.py

import logging
from .base_agent import BaseAgent
from core.edgar_direct_manager import EdgarDirectManager

logger = logging.getLogger(__name__)

class ReportPart1Agent(BaseAgent):
    def __init__(self, llm, unstructured_agent, company_name: str, status_placeholder=None):
        super().__init__(name="ReportPart1Agent", llm=llm)
        self.goal = "Generate Part I of the report: General Presentation."
        self.backstory = "You are a financial analyst tasked with compiling general company information."
        self.unstructured_agent = unstructured_agent
        self.company_name = company_name
        self.status_placeholder = status_placeholder

        self.edgar_manager = EdgarDirectManager(
            ticker_symbol=self.company_name,
            user_agent="Vincent Cotella <vincent.cotella@edu.devinci.fr>",
            report_type="10-K"
        )

    def generate_response(self, use_rag: bool, context_dict: dict) -> dict:
        if self.status_placeholder:
            self.status_placeholder.info("ReportPart1Agent: Starting Part I generation...")

        subtasks = {
            "Ticker": f"Retrieve the ticker symbol of {self.company_name}.",
            "Name": f"Confirm the full company name for {self.company_name}.",
            "Country": f"Identify the country of domicile for {self.company_name}.",
            "Sector": f"Determine the business sector or industry of {self.company_name}.",
            "Description": f"Provide a concise description of {self.company_name} from the 10-K.",
            "CEO": f"Find the name of the CEO of {self.company_name}.",
            "Headquarters": f"Locate the headquarters of {self.company_name}.",
            "Employees": f"Retrieve the number of employees of {self.company_name}."
        }

        # Mapping subtask -> liste d'Items à concaténer
        # Par exemple, on prend seulement 2 ou 3 Items, pas 1..4 en entier, 
        # pour éviter de dépasser les limites du modèle.
        item_mapping = {
            "Ticker":       ["Item 1"],
            "Name":         ["Item 1"],
            "Country":      ["Item 1"],
            "Sector":       ["Item 1", "Item 1A"],  # Ex, si Info sur sector dans Risk Factors
            "Description":  ["Item 1"],  # ou ["Item 1", "Item 1A"] si besoin
            "CEO":          ["Item 1"],
            "Headquarters": ["Item 1"],
            "Employees":    ["Item 1"]
        }

        results = {}

        for field, query in subtasks.items():
            if self.status_placeholder:
                self.status_placeholder.info(f"ReportPart1Agent: Processing {field}...")

            if use_rag:
                # RAG => base vectorielle
                chunks = self.unstructured_agent.generate_response(query)
                context_data = " ".join(chunks)
            else:
                # RAW => on concatène les Items spécifiques à ce subtask
                items_to_join = item_mapping.get(field, [])
                context_data = self.edgar_manager.get_items_concat(items_to_join)

            logger.debug(f"[DEBUG] Context for '{field}' => {context_data[:300]}...")

            prompt = (
                f"{self.backstory}\n\n"
                f"Goal: Extract the {field} of {self.company_name}.\n\n"
                f"Context:\n{context_data}\n\n"
                f"Please provide the {field} based on the data above.\n"
                f"If not available, respond with 'Not Available'."
            )

            response = self.llm(prompt).strip()
            results[field] = response

        if self.status_placeholder:
            self.status_placeholder.success("ReportPart1Agent: Part I generation completed.")

        return results
