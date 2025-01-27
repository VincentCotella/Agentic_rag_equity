# app.py

import os
import re
import json
import logging
import traceback
import streamlit as st
import pandas as pd
import requests

from config import GROQ_API_KEY, MODEL_NAME
from core.groq_llm import GROQLLM
from core.multi_agentic_rag import MultiAgenticRAG

# Pour la partie DataFrame & Plot
import plotly.express as px
import plotly.graph_objs as go
from io import BytesIO

# Tools/Agents LLM côté CSV
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from groq import Groq  # si nécessaire
import pandas as pd

# RAG data_management: base Chroma
from core.data_management import (
    clear_database,
    list_chroma_documents,
    add_custom_documents
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# FONCTIONS SPÉCIFIQUES À LA PARTIE CSV / PLOT
# --------------------------------------------------------------------------------

def load_dataframe(uploaded_file):
    """Charge un fichier CSV dans un DataFrame Pandas."""
    return pd.read_csv(uploaded_file)

def display_dataframe_info(df):
    """Affiche un aperçu du DataFrame."""
    st.write("### Aperçu de vos données")
    st.write(df.head())
    st.write("### Colonnes disponibles")
    st.write(df.columns.tolist())

def filter_dataframe_columns(df):
    """Permet à l'utilisateur de sélectionner des colonnes à afficher / manipuler."""
    selected_columns = st.multiselect("Sélectionnez les colonnes à inclure", df.columns.tolist())
    return df[selected_columns] if selected_columns else df

def extract_python_code(text):
    """Extrait le bloc de code python encapsulé entre ```python ...```."""
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        st.warning("Aucun code Python trouvé dans la réponse.", icon="🚨")
        return None
    return matches[0].strip()

def generate_plot_tool(query, df):
    """Tool pour générer un plot en utilisant Plotly, basé sur la requête utilisateur."""
    client = Groq()
    code_prompt = (
        "You are an AI assistant specialized in analyzing financial data. "
        "Your task is to generate clear and insightful plots based on the user query. "
        "The solution should strictly use Plotly or Plotly Express and no other plotting library. "
        "You should based your answer on the data provide by the user."
        "Return the Python code in a fenced code block: ```python <code>```. "
        "No other plotting library allowed."
    )

    # On appelle le modèle
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{"role": "user", "content": code_prompt + query}],
        temperature=0.3,
    )
    # Extraction du code
    code = extract_python_code(response.choices[0].message.content)
    if code is None:
        return "Aucun code Plotly valide n'a été généré."

    # Supprime un éventuel fig.show() car on va l'afficher via st.plotly_chart
    code = code.replace("fig.show()", "")
    code += "\nst.plotly_chart(fig, theme='streamlit', use_container_width=True)"

    # On exécute le code
    try:
        local_vars = {"df": df, "st": st, "px": px, "go": go}
        exec(code, local_vars)
        return "Plot generated successfully."
    except Exception as e:
        return f"Erreur lors de l'exécution du code généré: {e}"

def get_recent_data_prompt():
    """Prompt pour analyser uniquement les données récentes."""
    return (
        "You are an AI assistant specialized in analyzing recent financial data. "
        "Focus on the two most recent years in the 'Date' column. "
        "Use Plotly or Plotly Express only. Return the code in a fenced block: ```python ...```"
    )

def get_historical_data_prompt():
    """Prompt pour analyser uniquement les données historiques (hors dernières années)."""
    return (
        "You are an AI assistant specialized in historical financial data analysis. "
        "Exclude the most recent year's data of the dataframe entirely and focus on older data only. "
        "Your task is to generate clear and informative plots based on user queries. "
        "You should exclude the most recent year's data entirely from your analysis and focus on long-term trends or historical context."
        "Avoid focusing on recent data and prioritize the historical context. "
        "You should based your answer on the data provide by the user."
        "Avoid using historical trends and concentrate solely on the current snapshot of the data. "
        "Use Plotly or Plotly Express only. Return the code in a fenced block: ```python ...```"
    )

# --------------------------------------------------------------------------------
# APPLICATION PRINCIPALE
# --------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="10-K & Financial Data Analysis", layout="wide")

    st.title("10-K Report Generator (Unstructured Data) + CSV Analysis")

    # =========================
    # SIDEBAR: Chroma DB options
    # =========================
    st.sidebar.header("Database Options")

    if st.sidebar.button("Reset Chroma DB"):
        with st.spinner("Resetting the database..."):
            try:
                clear_database()
                st.sidebar.success("Chroma database has been reset (deleted).")
            except Exception as e:
                st.sidebar.error(f"Error while resetting DB: {e}")
                logger.error(traceback.format_exc())
                st.stop()

    if st.sidebar.button("List Chroma Documents"):
        docs_info = list_chroma_documents()
        if not docs_info:
            st.sidebar.info("No documents found in Chroma DB.")
        else:
            st.sidebar.write(f"Found {len(docs_info)} documents/chunks:")
            for doc in docs_info:
                st.sidebar.write(f"- ID: {doc['id']}")
                st.sidebar.write(f"  Source: {doc['metadata']['source']}")
                st.sidebar.write(f"  Snippet: {doc['content_snippet']}")

    # Upload custom docs to Chroma
    st.sidebar.write("## Add Custom Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose .txt or .md files to embed in Chroma",
        accept_multiple_files=True,
        type=["txt", "md"]
    )
    if uploaded_files:
        if st.sidebar.button("Add to DB"):
            try:
                temp_dir = "temp_uploads"
                os.makedirs(temp_dir, exist_ok=True)
                file_paths = []
                for uf in uploaded_files:
                    temp_path = os.path.join(temp_dir, uf.name)
                    with open(temp_path, "wb") as f:
                        f.write(uf.getvalue())
                    file_paths.append(temp_path)

                add_custom_documents(file_paths)
                st.sidebar.success(f"Successfully added {len(file_paths)} file(s) to Chroma DB!")
            except Exception as e:
                st.sidebar.error(f"Error while adding files to DB: {e}")
                logger.error(traceback.format_exc())

    # =========================
    # SECTION: 10-K REPORT
    # =========================
    st.header("Generate the 10-K Report")
    ticker_symbol = st.text_input("Enter the company's ticker symbol (e.g., 'AAPL'):", "")

    if st.button("Initialize System"):
        if not ticker_symbol:
            st.warning("Please enter a ticker symbol before initializing.")
        else:
            with st.spinner("Initializing the system..."):
                try:
                    llm = GROQLLM(api_key=GROQ_API_KEY, model=MODEL_NAME)
                    rag = MultiAgenticRAG(llm=llm, company_name=ticker_symbol)
                    st.session_state["rag"] = rag
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {e}")
                    logger.error(traceback.format_exc())
                    st.stop()

    if "rag" in st.session_state:
        rag = st.session_state["rag"]
        context_choice = st.radio(
            "Select context source for each part:",
            ["Use RAG (vector DB)", "Use raw 10-K sections"]
        )
        use_rag = (context_choice == "Use RAG (vector DB)")

        st.subheader("Generate Report Parts")

        if st.button("Generate Part I"):
            try:
                with st.spinner("Generating Part I..."):
                    part1 = rag.generate_report_part1(use_rag=use_rag)
                    st.subheader("Part I: General Presentation")
                    st.json(part1)
                    json_str = json.dumps(part1, indent=4)
                    st.download_button("Download Part I JSON", data=json_str, file_name="part1.json")
            except Exception as e:
                st.error(f"Error generating Part I: {e}")
                logger.error(traceback.format_exc())

        if st.button("Generate Part II"):
            try:
                with st.spinner("Generating Part II..."):
                    part2 = rag.generate_report_part2(use_rag=use_rag)
                    st.subheader("Part II: Key Financial Figures")
                    st.json(part2)
                    json_str = json.dumps(part2, indent=4)
                    st.download_button("Download Part II JSON", data=json_str, file_name="part2.json")
            except Exception as e:
                st.error(f"Error generating Part II: {e}")
                logger.error(traceback.format_exc())

        if st.button("Generate Part III"):
            try:
                with st.spinner("Generating Part III..."):
                    part3 = rag.generate_report_part3(use_rag=use_rag)
                    st.subheader("Part III: Annual Performance Analysis")
                    st.json(part3)
                    json_str = json.dumps(part3, indent=4)
                    st.download_button("Download Part III JSON", data=json_str, file_name="part3.json")
            except Exception as e:
                st.error(f"Error generating Part III: {e}")
                logger.error(traceback.format_exc())

        if st.button("Generate Part IV"):
            try:
                with st.spinner("Generating Part IV..."):
                    part4 = rag.generate_report_part4(use_rag=use_rag)
                    st.subheader("Part IV: Market Position and Competitors")
                    st.json(part4)
                    json_str = json.dumps(part4, indent=4)
                    st.download_button("Download Part IV JSON", data=json_str, file_name="part4.json")
            except Exception as e:
                st.error(f"Error generating Part IV: {e}")
                logger.error(traceback.format_exc())

        if st.button("Generate Part V"):
            try:
                with st.spinner("Generating Part V..."):
                    part5 = rag.generate_report_part5(use_rag=use_rag)
                    st.subheader("Part V: Risks and Challenges")
                    st.json(part5)
                    json_str = json.dumps(part5, indent=4)
                    st.download_button("Download Part V JSON", data=json_str, file_name="part5.json")
            except Exception as e:
                st.error(f"Error generating Part V: {e}")
                logger.error(traceback.format_exc())
    else:
        st.info("Please initialize the system first (enter ticker & click 'Initialize System').")

    # =========================
    # SECTION: FINANCIAL CSV DATA ANALYSIS
    # =========================
    st.header("Financial CSV Data Analysis")

    # Upload CSV
    uploaded_csv = st.file_uploader("Upload a CSV for financial data analysis", type=["csv"])

    if uploaded_csv:
        df = load_dataframe(uploaded_csv)
        display_dataframe_info(df)
        filtered_df = filter_dataframe_columns(df)

        # Configuration du LLM Groq
        st.sidebar.header("Configuration du LLM pour CSV Analysis")
        model_name_csv = "llama-3.1-8b-instant"  # ou un autre
        max_tokens_csv = st.sidebar.slider("Max tokens (CSV LLM)", min_value=100, max_value=2000, value=500)

        llm_csv = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=model_name_csv,
            temperature=0.3,
            max_tokens=max_tokens_csv,
        )

        # Création d'agents pour data analysis
        recent_prompt = get_recent_data_prompt()
        historical_prompt = get_historical_data_prompt()

        recent_data_tool = Tool(
            name="Generate Plot (Recent Data)",
            func=lambda query: generate_plot_tool(query, filtered_df),
            description=recent_prompt
        )
        historical_data_tool = Tool(
            name="Generate Plot (Historical Data)",
            func=lambda query: generate_plot_tool(query, filtered_df),
            description=historical_prompt
        )

        pandas_df_recent_agent = create_pandas_dataframe_agent(
            llm_csv,
            filtered_df,
            extra_tools=[recent_data_tool],
            verbose=True,
            max_iterations=20,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
        )
        pandas_df_historical_agent = create_pandas_dataframe_agent(
            llm_csv,
            filtered_df,
            extra_tools=[historical_data_tool],
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=20,
        )

        st.write("### Agents disponibles pour générer des graphes à partir de votre CSV")
        agent_option = st.selectbox(
            "Choisissez un agent",
            ["Analyse de données récentes", "Analyse des données historiques"]
        )

        query = st.text_input("Entrez une requête pour générer un plot à partir du CSV :")

        if query:
            if agent_option == "Analyse de données récentes":
                with st.spinner("Génération du plot (données récentes)..."):
                    response = pandas_df_recent_agent.invoke(query)
                    st.write("### Résultat / Plot")
                    st.write(response)
            else:
                with st.spinner("Génération du plot (données historiques)..."):
                    response = pandas_df_historical_agent.invoke(query)
                    st.write("### Résultat / Plot")
                    st.write(response)


    # =========================
    # TEST GROQ API
    # =========================
    st.header("Other Actions")
    if st.button("Test GROQ API"):
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models = response.json()
            st.success("GROQ API is working correctly.")
            st.write(models)
        except Exception as e:
            st.error(f"Error calling GROQ API: {e}")
            logger.error(traceback.format_exc())

    # =========================
    # STRUCTURED PRIVATE EQUITY DATA
    # =========================
    st.header("Structured Private Equity Data")
    st.write("No structured data available as per the current configuration.")


if __name__ == "__main__":
    main()
