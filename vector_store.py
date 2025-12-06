"""Module for managing the vector store."""
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import DB_DIR, EMBEDDING_MODEL


def create_or_load_vector_store(documents: list, db_path: Path = DB_DIR):
    """
    Create a new vector store or load an existing one.
    
    Args:
        documents: List of documents to add to the store (if creating new)
        db_path: Path to store the vector database
        
    Returns:
        Chroma vector store instance
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Check if database already exists
    if list(db_path.glob("*")):
        print("Loading existing vector store...\n")
        vector_store = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings
        )
    else:
        print("Creating new vector store...")
        if not documents:
            raise ValueError("Documents required to create new vector store")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(db_path)
        )
        vector_store.persist()
        print("Vector store created and persisted\n")
    
    return vector_store


def update_vector_store(documents: list, db_path: Path = DB_DIR):
    """
    Update an existing vector store with new documents.
    
    Args:
        documents: List of new documents to add
        db_path: Path to the vector database
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents)
    vector_store.persist()
    print(f"Added {len(documents)} documents to vector store\n")
    return vector_store
