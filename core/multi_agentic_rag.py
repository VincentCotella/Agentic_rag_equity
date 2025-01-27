# core/multi_agentic_rag.py

import logging

from agents.unstructured_data_agent import UnstructuredDataAgent
from agents.report_part1_agent import ReportPart1Agent
from agents.report_part2_agent import ReportPart2Agent
from agents.report_part3_agent import ReportPart3Agent
from agents.report_part4_agent import ReportPart4Agent
from agents.report_part5_agent import ReportPart5Agent

logger = logging.getLogger(__name__)

class MultiAgenticRAG:
    def __init__(self, llm, company_name: str, status_placeholder=None):
        """
        Initialise le multi-agent RAG.
        """
        self.llm = llm
        self.company_name = company_name.upper()
        self.status_placeholder = status_placeholder

        # Unstructured agent (RAG)
        self.unstructured_agent = UnstructuredDataAgent()

        # On laisse un context_dict vide ou minimal. 
        # (Certains agents l'ignorent s'ils sont en mode RAW.)
        self.context_dict = {}

        # Initialize report part agents
        self.report_part1_agent = ReportPart1Agent(
            llm=self.llm,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part2_agent = ReportPart2Agent(
            llm=self.llm,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part3_agent = ReportPart3Agent(
            llm=self.llm,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part4_agent = ReportPart4Agent(
            llm=self.llm,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part5_agent = ReportPart5Agent(
            llm=self.llm,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )

    def generate_report_part1(self, use_rag: bool):
        return self.report_part1_agent.generate_response(
            use_rag=use_rag,
            context_dict=self.context_dict
        )

    def generate_report_part2(self, use_rag: bool):
        return self.report_part2_agent.generate_response(
            use_rag=use_rag,
            context_dict=self.context_dict
        )

    def generate_report_part3(self, use_rag: bool):
        return self.report_part3_agent.generate_response(
            use_rag=use_rag,
            context_dict=self.context_dict
        )

    def generate_report_part4(self, use_rag: bool):
        return self.report_part4_agent.generate_response(
            use_rag=use_rag,
            context_dict=self.context_dict
        )

    def generate_report_part5(self, use_rag: bool):
        return self.report_part5_agent.generate_response(
            use_rag=use_rag,
            context_dict=self.context_dict
        )
