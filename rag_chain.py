"""Module for the RAG chain."""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from config import LLM_MODEL
import re


def create_rag_chain(vector_store):
    """
    Create a RAG chain for question answering.
    
    Args:
        vector_store: Chroma vector store with document embeddings
        
    Returns:
        RetrievalQA chain instance
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.7,
    )
    
    # Custom prompt template
    prompt_template = """I am the owner of this diary. Use the following pieces of context from the diaries to answer the question. 
If you don't find relevant information in the context, say you don't have that information in the diaries.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    
    return qa_chain


def query_rag_chain(qa_chain, question: str):
    """
    Query the RAG chain and return the answer with sources.
    
    Args:
        qa_chain: The RAG chain instance
        question: The question to ask
        
    Returns:
        Dictionary with answer and deduplicated source documents
    """
    result = qa_chain({"query": question})

    # Deduplicate source documents while preserving order.
    # Use (source, page, date) as the dedupe key when available.
    seen = set()
    unique_docs = []
    for doc in result.get("source_documents", []):
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        date = doc.metadata.get("date", None)

        key = (src, page, date)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return {
        "answer": result.get("result"),
        "source_documents": unique_docs
    }


def print_query_result(result: dict):
    """
    Pretty print the query result with answer and context.
    
    Args:
        result: Dictionary containing answer and source documents
    """
    print(f"Answer:\n{result['answer']}\n")
    print("=" * 60)
    print("Context from Diaries:")
    print("=" * 60 + "\n")
    
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        date = doc.metadata.get('date', 'N/A')
        content = doc.page_content
        
        print(f"[Source {i}] {source} - Page {page} - Date: {date}")
        print("-" * 60)
        print(f"{content}")
        print("-" * 60 + "\n")
