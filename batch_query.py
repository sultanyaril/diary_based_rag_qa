"""Module for batch querying (non-interactive mode)."""
from document_loader import load_pdf_documents, split_documents_by_date
from vector_store import create_or_load_vector_store
from rag_chain import create_rag_chain, query_rag_chain


def batch_query(questions: list):
    """
    Answer multiple questions in batch mode.
    
    Args:
        questions: List of questions to answer
        
    Returns:
        List of results with answers and sources
    """
    # Initialize system
    print("Initializing system...")
    documents = load_pdf_documents()
    split_docs = split_documents_by_date(documents)
    vector_store = create_or_load_vector_store(split_docs)
    qa_chain = create_rag_chain(vector_store)
    
    print(f"Processing {len(questions)} question(s)...\n")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        result = query_rag_chain(qa_chain, question)
        results.append({
            "question": question,
            "answer": result["answer"],
            "sources": [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
        })
        print(f"Answer: {result['answer'][:100]}...\n")
    
    return results


if __name__ == "__main__":
    # Example batch queries
    questions = [
        "What are the main events mentioned in the diaries?",
        "What emotions or feelings are expressed most frequently?",
        "Are there any recurring themes or patterns?"
    ]
    
    results = batch_query(questions)
