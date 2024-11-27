# core/multi_agentic_rag.py

from agents.structured_data_agent import StructuredDataAgent
from agents.unstructured_data_agent import UnstructuredDataAgent
from agents.report_part1_agent import ReportPart1Agent
from agents.report_part2_agent import ReportPart2Agent
from agents.report_part3_agent import ReportPart3Agent
from agents.report_part4_agent import ReportPart4Agent
from agents.report_part5_agent import ReportPart5Agent

from config import MODEL_NAME
from core.data_management import load_structured_data
from core.groq_llm import GROQLLM

class MultiAgenticRAG:
    def __init__(self, llm: GROQLLM, company_name: str, status_placeholder=None):
        """
        Initialize the multi-agent RAG system with structured data.

        Args:
            llm (GROQLLM): The language model to use.
            company_name (str): The name of the company.
            status_placeholder: Streamlit placeholder for status messages.
        """
        self.llm = llm
        self.company_name = company_name
        self.status_placeholder = status_placeholder

        # Load structured data
        structured_data_df = load_structured_data()

        # Initialize the data agents
        self.structured_agent = StructuredDataAgent(structured_data=structured_data_df)
        self.unstructured_agent = UnstructuredDataAgent()

        # Initialize the report part agents with access to data agents and company_name
        self.report_part1_agent = ReportPart1Agent(
            llm=self.llm,
            structured_agent=self.structured_agent,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part2_agent = ReportPart2Agent(
            llm=self.llm,
            structured_agent=self.structured_agent,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part3_agent = ReportPart3Agent(
            llm=self.llm,
            structured_agent=self.structured_agent,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part4_agent = ReportPart4Agent(
            llm=self.llm,
            structured_agent=self.structured_agent,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        self.report_part5_agent = ReportPart5Agent(
            llm=self.llm,
            structured_agent=self.structured_agent,
            unstructured_agent=self.unstructured_agent,
            company_name=self.company_name,
            status_placeholder=self.status_placeholder
        )
        # ... Initialize other agents as needed

    def generate_report(self, progress_callback=None) -> dict:
        """
        Orchestrate the generation of the report by calling each report part agent.

        Args:
            progress_callback: Function to call to update progress.

        Returns:
            dict: The final report containing all parts.
        """
        report = {}

        # List of report part agents
        agents = [
            ("Part I", self.report_part1_agent),
            ("Part II", self.report_part2_agent),
            ("Part III", self.report_part3_agent),
            ("Part IV", self.report_part4_agent),
            ("Part V", self.report_part5_agent),
            # ... Include other parts as needed
        ]

        total_agents = len(agents)
        for idx, (part_name, agent) in enumerate(agents):
            if self.status_placeholder:
                self.status_placeholder.info(f"Generating {part_name}...")
            part_content = agent.generate_response()
            report[part_name] = part_content
            if progress_callback:
                progress_callback()
            if self.status_placeholder:
                self.status_placeholder.empty()

        if self.status_placeholder:
            self.status_placeholder.success("Report generation completed.")

        return report

