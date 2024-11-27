# data_management.py

import os
import shutil
import uuid
import logging
import unicodedata
import re
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from embedding import get_embedding_function
from config import CHROMA_PATH, DATA_PATH, STRUCTURED_DATA_CSV_PATH

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_documents():
    """
    Récupère la liste des documents PDF disponibles dans le répertoire DATA_PATH.

    Returns:
        list: Liste des noms de fichiers PDF dans DATA_PATH.
    """
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            documents.append(filename)
    return documents

def load_documents(documents_to_process=None):
    """
    Charge les fichiers PDF depuis le répertoire spécifié dans DATA_PATH.

    Args:
        documents_to_process (list or str): Liste des noms de fichiers à traiter,
                                             ou "all" pour traiter tous les documents.

    Returns:
        list: Liste des documents chargés.
    """
    all_documents = []
    document_filenames = []

    if documents_to_process == "all" or documents_to_process is None:
        document_filenames = get_available_documents()
    else:
        document_filenames = documents_to_process

    for filename in document_filenames:
        file_path = os.path.join(DATA_PATH, filename)
        if os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_documents.extend(documents)
                logger.info(f"Loaded document: {filename}")
            except Exception as e:
                logger.error(f"Error loading document {filename}: {e}")
        else:
            logger.warning(f"File {filename} does not exist in {DATA_PATH}.")

    return all_documents

def clean_text(text):
    """
    Nettoie le texte en supprimant les caractères problématiques et en normalisant l'encodage.

    Args:
        text (str): Le texte à nettoyer.

    Returns:
        str: Le texte nettoyé.
    """
    # Supprimer les caractères de contrôle et non assignés
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C')

    # Remplacer les espaces multiples par un seul espace
    text = re.sub(r'\s+', ' ', text)

    # Normaliser le texte (forme NFKD)
    text = unicodedata.normalize('NFKD', text)

    # Optionnel : Supprimer les caractères spéciaux indésirables
    # text = re.sub(r'[^\w\s.,!?;:()\'"-]', '', text)

    return text.strip()

def split_documents(documents):
    """
    Divise chaque document en segments pour un meilleur indexage et une meilleure récupération.
    Le texte de chaque document est nettoyé avant le découpage.

    Args:
        documents (list): Liste des documents à découper.

    Returns:
        list: Liste des chunks.
    """
    cleaned_documents = []
    for doc in documents:
        # Nettoyer le contenu du document
        cleaned_content = clean_text(doc.page_content)
        # Créer un nouveau Document avec le contenu nettoyé
        cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata)
        cleaned_documents.append(cleaned_doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(cleaned_documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

def calculate_chunk_ids(chunks):
    """
    Génère un ID unique pour chaque segment de document.

    Args:
        chunks (list): Liste des chunks à traiter.

    Returns:
        list: Liste des chunks avec IDs mis à jour.
    """
    for chunk in chunks:
        if 'id' not in chunk.metadata:
            chunk.metadata["id"] = str(uuid.uuid4())
    return chunks

def add_to_chroma(chunks, progress_callback=None):
    """
    Ajoute les segments de documents (chunks) à la base de données Chroma.

    Args:
        chunks (list): Liste des chunks à ajouter.
        progress_callback (function, optional): Fonction pour mettre à jour la progression.
    """
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
    except Exception as e:
        logger.error(f"Error initializing Chroma database: {e}")
        raise

    # Générer des IDs pour les chunks
    chunks_with_ids = calculate_chunk_ids(chunks)
    total_chunks = len(chunks_with_ids)

    # Préparation pour le suivi de la progression
    if progress_callback:
        progress_callback(0.0, f"Adding {total_chunks} chunks to the database...")

    # Taille du batch pour l'ajout des chunks
    batch_size = 50  # Ajustez si nécessaire

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks_with_ids[i:i+batch_size]
        texts = [chunk.page_content for chunk in batch_chunks]
        metadatas = [chunk.metadata for chunk in batch_chunks]
        ids = [chunk.metadata["id"] for chunk in batch_chunks]

        # Nettoyer les textes du batch (si ce n'est pas déjà fait)
        cleaned_texts = [clean_text(text) for text in texts]

        # Vérifier si les textes ne sont pas vides après nettoyage
        valid_indices = [idx for idx, text in enumerate(cleaned_texts) if text.strip()]
        if not valid_indices:
            logger.warning(f"All chunks in batch starting at index {i} are empty after cleaning. They will be skipped.")
            continue

        # Filtrer les données valides
        batch_chunks = [batch_chunks[idx] for idx in valid_indices]
        cleaned_texts = [cleaned_texts[idx] for idx in valid_indices]
        metadatas = [metadatas[idx] for idx in valid_indices]
        ids = [ids[idx] for idx in valid_indices]

        try:
            # Ajouter à la base de données
            db.add_texts(texts=cleaned_texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Error adding documents to Chroma at batch starting with index {i}: {e}")
            if progress_callback:
                progress_callback((i + len(batch_chunks)) / total_chunks, f"Error adding chunks to database at chunk {i}")
            raise

        # Mise à jour de la progression
        if progress_callback:
            progress = (i + len(batch_chunks)) / total_chunks
            progress_callback(progress, f"Adding chunks to database: {i + len(batch_chunks)}/{total_chunks}")

    logger.info(f"Added {total_chunks} chunks to the database.")

def initialize_database(progress_callback=None, documents_to_process=None):
    """
    Initialise la base de données en chargeant les documents, les découpant en segments,
    et en ajoutant ces segments à Chroma.

    Args:
        progress_callback (function, optional): Fonction pour mettre à jour la progression.
        documents_to_process (list or str, optional): Liste des documents à traiter ou "all".
    """
    total_steps = 3  # Nombre d'étapes dans cette fonction
    current_step = 0

    try:
        # Étape 1 : Charger les documents
        if progress_callback:
            progress_callback(current_step / total_steps, "Loading documents...")
        documents = load_documents(documents_to_process)
        logger.info("Documents loaded.")
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Splitting documents...")

        # Étape 2 : Diviser les documents en chunks
        chunks = split_documents(documents)
        logger.info(f"{len(chunks)} chunks created.")
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Adding chunks to Chroma database...")

        # Étape 3 : Ajouter les chunks à la base de données Chroma
        add_to_chroma(chunks, progress_callback=progress_callback)
        logger.info("Chunks added to Chroma.")
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps, "Database initialization complete.")
        logger.info("initialize_database completed.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

def clear_database():
    """
    Supprime la base de données Chroma en supprimant tous les fichiers dans le répertoire CHROMA_PATH.
    """
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            logger.info(f"Directory '{CHROMA_PATH}' successfully deleted.")
        except PermissionError as e:
            logger.error(f"Permission error when deleting directory '{CHROMA_PATH}': {e}")
            logger.error("Ensure no process is using the Chroma database.")
            raise

def load_structured_data():
    """
    Charge et formate les données de private equity à partir d'un fichier CSV.

    Returns:
        pandas.DataFrame: Les données chargées sous forme de DataFrame.
    """
    csv_path = STRUCTURED_DATA_CSV_PATH
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Structured data loaded from {csv_path}.")
    except FileNotFoundError:
        logger.error(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame()  # Retourne un DataFrame vide si le fichier n'est pas trouvé
    except Exception as e:
        logger.error(f"Error loading structured data: {e}")
        return pd.DataFrame()
    return df
