# core/data_management.py

import os
import shutil
import uuid
import logging
from typing import List, Dict, Any

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from embedding import get_embedding_function
from config import CHROMA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_database():
    """
    Supprime le répertoire CHROMA_PATH pour réinitialiser la base Chroma.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"Répertoire Chroma supprimé: {CHROMA_PATH}")
    else:
        logger.info("Aucune base Chroma à supprimer (répertoire introuvable).")

def list_chroma_documents() -> List[Dict[str, Any]]:
    """
    Liste tous les documents (chunks) déjà indexés dans la base Chroma.
    """
    if not os.path.exists(CHROMA_PATH):
        logger.warning(f"Aucune base Chroma trouvée dans {CHROMA_PATH}.")
        return []
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    store = db.get()  # returns dict with keys: ["ids", "embeddings", "metadatas", "documents"]
    docs_info = []
    for i, doc_content in enumerate(store["documents"]):
        doc_id = store["ids"][i]
        meta = store["metadatas"][i]
        snippet = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
        docs_info.append({
            "id": doc_id,
            "metadata": meta,
            "content_snippet": snippet
        })
    return docs_info

def add_custom_documents(file_paths: List[str]):
    """
    Ajoute manuellement une liste de fichiers .txt / .md à la base Chroma.
    """
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info("Création du répertoire Chroma...")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    # On lit chaque fichier, on le split en chunks, on indexe
    all_chunks = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            logger.warning(f"Fichier introuvable: {fp}")
            continue
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(page_content=text, metadata={"source": os.path.basename(fp)})
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
        # ou .split_documents([doc]) selon la version
        for c in chunks:
            chunk_doc = Document(page_content=c, metadata={"source": os.path.basename(fp)})
            all_chunks.append(chunk_doc)

    texts = [d.page_content for d in all_chunks]
    metadatas = [{"source": d.metadata["source"]} for d in all_chunks]
    ids = [str(uuid.uuid4()) for _ in all_chunks]
    db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    db.persist()
    logger.info(f"{len(file_paths)} fichier(s) ajouté(s). Total {len(all_chunks)} chunks insérés.")
