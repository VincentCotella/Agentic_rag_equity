# app.py

import streamlit as st
import os
import requests
import json
import logging
import traceback

from config import GROQ_API_KEY, MODEL_NAME
from core.groq_llm import GROQLLM
from core.multi_agentic_rag import MultiAgenticRAG

# On importe les fonctions utiles pour gérer la base Chroma (RAG)
# Note: on ne fait plus "from core.data_management import initialize_database_with_api"
from core.data_management import (
    clear_database,
    list_chroma_documents,
    add_custom_documents
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("10-K Report Generator (Unstructured Data Only)")

    # -------------------------------------------------------------------------
    # SIDEBAR - Options sur la base Chroma (RAG)
    # -------------------------------------------------------------------------
    st.sidebar.header("Database Options")

    # 1) Bouton pour Reset complet de Chroma
    if st.sidebar.button("Reset Chroma DB"):
        with st.spinner("Resetting the database..."):
            try:
                clear_database()
                st.sidebar.success("Chroma database has been reset (deleted).")
            except Exception as e:
                st.sidebar.error(f"Error while resetting DB: {e}")
                logger.error(traceback.format_exc())
                st.stop()

    # 2) Bouton pour lister les documents existants
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

    # 3) Téléversement de fichiers à ajouter manuellement dans Chroma
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

    # -------------------------------------------------------------------------
    # MAIN LAYOUT - Génération du rapport 10-K
    # -------------------------------------------------------------------------
    st.header("Generate the 10-K Report")

    # Saisie du ticker
    ticker_symbol = st.text_input("Enter the company's ticker symbol (e.g., 'AAPL'):", "")

    if st.button("Initialize System"):
        if not ticker_symbol:
            st.warning("Please enter a ticker symbol before initializing.")
        else:
            with st.spinner("Initializing the system..."):
                try:
                    # Ici, on ne fait plus initialize_database_with_api(ticker_symbol)
                    # car la fonction a été retirée (ou est un stub).
                    # On se contente de créer LLM + RAG :

                    llm = GROQLLM(api_key=GROQ_API_KEY, model=MODEL_NAME)
                    rag = MultiAgenticRAG(llm=llm, company_name=ticker_symbol)
                    st.session_state["rag"] = rag
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Error initializing system: {e}")
                    logger.error(traceback.format_exc())
                    st.stop()

    # Vérifier si on a un RAG dans session_state
    if "rag" in st.session_state:
        rag = st.session_state["rag"]

        context_choice = st.radio(
            "Select context source for each part:",
            ["Use RAG (vector DB)", "Use raw 10-K sections"]
        )
        use_rag = (context_choice == "Use RAG (vector DB)")

        st.subheader("Generate Report Parts")

        # PART I
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

        # PART II
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

        # PART III
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

        # PART IV
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

        # PART V
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

    # -------------------------------------------------------------------------
    # TEST GROQ API
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # SECTION - Structured Private Equity Data (placeholder)
    # -------------------------------------------------------------------------
    st.header("Structured Private Equity Data")
    st.write("No structured data available as per the current configuration.")


if __name__ == "__main__":
    main()
