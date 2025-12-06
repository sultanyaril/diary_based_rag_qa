"""Main module for the Diary-based RAG QA system."""
import sys
from pathlib import Path
from document_loader import load_pdf_documents, split_documents_by_date
from vector_store import create_or_load_vector_store, update_vector_store
from rag_chain import create_rag_chain, query_rag_chain, print_query_result
from config import DATA_DIR


def initialize_system():
    """Initialize the RAG system."""
    print("=" * 60)
    print("Diary-based RAG Question Answering System")
    print("=" * 60 + "\n")
    
    # Load and process documents
    documents = load_pdf_documents()
    split_docs = split_documents_by_date(documents)
    
    # Create or load vector store
    vector_store = create_or_load_vector_store(split_docs)
    
    # Create RAG chain
    qa_chain = create_rag_chain(vector_store)
    
    print("System initialized successfully!\n")
    return qa_chain


def interactive_qa(qa_chain):
    """Run interactive question answering session."""
    print("Enter your questions about the diaries (type 'exit' to quit):\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            if not question:
                print("Please enter a valid question.\n")
                continue
            
            print("\nSearching diaries...\n")
            result = query_rag_chain(qa_chain, question)
            
            print_query_result(result)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main entry point."""
    try:
        qa_chain = initialize_system()
        interactive_qa(qa_chain)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure you have PDF files in the {DATA_DIR} directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
