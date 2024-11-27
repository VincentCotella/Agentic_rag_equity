# app.py

import streamlit as st
from core.data_management import (
    initialize_database,
    clear_database,
    load_structured_data,
    get_available_documents,
)
from core.multi_agentic_rag import MultiAgenticRAG
from core.groq_llm import GROQLLM
from config import GROQ_API_KEY, MODEL_NAME
import os
import requests
import json
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Multi-Agent Financial Analysis Assistant with GROQ")

    st.sidebar.header("Options")
    reset_db = st.sidebar.button("Reset Database")

    if reset_db:
        with st.spinner("Resetting the database..."):
            try:
                clear_database()
                st.success("Database reset successfully!")
                st.info("Please restart the application to apply changes.")
                st.stop()
            except Exception as e:
                st.error(f"Error resetting the database: {e}")
                logger.error(f"Error resetting the database: {e}")
                logger.error(traceback.format_exc())

    st.sidebar.subheader("Upload Documents")
    upload_pdf = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if upload_pdf:
        pdf_dir = "PDF"
        os.makedirs(pdf_dir, exist_ok=True)

        # Save uploaded PDFs
        uploaded_filenames = []
        for uploaded_file in upload_pdf:
            pdf_path = os.path.join(pdf_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_filenames.append(uploaded_file.name)
            st.sidebar.success(f"Uploaded file: {uploaded_file.name}")
            logger.info(f"Uploaded file: {uploaded_file.name}")

        # Allow user to select which documents to add
        st.sidebar.subheader("Select Documents to Add to Database")
        available_documents = get_available_documents()
        documents_to_process = st.sidebar.multiselect(
            "Select documents to add:",
            options=uploaded_filenames,
            default=uploaded_filenames  # By default, select all uploaded files
        )

        # "Process Documents" button
        if st.sidebar.button("Process Documents"):
            # Progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)

            try:
                status_text.text("Processing documents...")
                initialize_database(
                    progress_callback=progress_callback,
                    documents_to_process=documents_to_process
                )
                progress_bar.progress(1.0)
                status_text.text("Document processing completed successfully.")
                st.success("Documents have been added and indexed successfully!")
                logger.info("Document processing completed successfully.")
            except Exception as e:
                st.error(f"Error during document processing: {e}")
                st.error(traceback.format_exc())
                progress_bar.empty()
                status_text.empty()
                logger.error(f"Error during document processing: {e}")
                logger.error(traceback.format_exc())
                return
            finally:
                # Clear progress bar and status text after completion
                progress_bar.empty()
                status_text.empty()

    st.sidebar.subheader("Other Actions")
    if st.sidebar.button("Test GROQ API"):
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            models = response.json()
            st.sidebar.success("GROQ API is working correctly.")
            st.sidebar.write(models)
        except Exception as e:
            st.sidebar.error(f"Error calling GROQ API: {e}")
            logger.error(f"Error calling GROQ API: {e}")
            logger.error(traceback.format_exc())

    st.header("Generate the 10-K Report")

    # Input field for the company name
    company_name = st.text_input("Enter the company name:")

    # "Generate Report" button
    if st.button("Generate Report"):
        if company_name:
            # Placeholders for status messages and progress bar
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            total_steps = 5  # Adjust based on the number of report parts
            current_step = 0

            def report_progress_callback():
                nonlocal current_step
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress)

            try:
                with st.spinner("Generating the report..."):
                    llm = GROQLLM(api_key=GROQ_API_KEY, model=MODEL_NAME)
                    rag = MultiAgenticRAG(llm=llm, company_name=company_name, status_placeholder=status_placeholder)
                    # Generate the report
                    report = rag.generate_report(progress_callback=report_progress_callback)
                    # Display each part
                    for part_name, part_content in report.items():
                        st.subheader(part_name)
                        if isinstance(part_content, dict):
                            st.json(part_content)
                            json_str = json.dumps(part_content, indent=4)
                            st.download_button(
                                label=f"Download {part_name} as JSON",
                                data=json_str,
                                file_name=f"{part_name.replace(' ', '_')}.json",
                                mime="application/json"
                            )
                        else:
                            st.write(part_content)
                            st.download_button(
                                label=f"Download {part_name} as Text",
                                data=part_content,
                                file_name=f"{part_name.replace(' ', '_')}.txt",
                                mime="text/plain"
                            )
                        # Note: Removed the extra call to report_progress_callback() here
                    status_placeholder.success("Report generation completed.")
            except Exception as e:
                st.error(f"Error during report generation: {e}")
                st.error(traceback.format_exc())
                logger.error(f"Error during report generation: {e}")
                logger.error(traceback.format_exc())
                return
            finally:
                # Clear the progress bar and status placeholder
                progress_bar.empty()
                status_placeholder.empty()
        else:
            st.warning("Please enter the company name before generating the report.")

    st.header("Structured Private Equity Data")
    structured_data = load_structured_data()
    if not structured_data.empty:
        st.dataframe(structured_data)
    else:
        st.write("No structured data available.")

if __name__ == "__main__":
    main()
