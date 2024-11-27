# embedding.py

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Initialise la fonction d'embedding pour transformer les documents en vecteurs
    utilisables par la base de données Chroma.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    return embeddings