"""Module for loading and processing PDF documents."""
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from config import DATA_DIR


def load_pdf_documents(data_dir: Path = DATA_DIR) -> list:
    """
    Load all PDF documents from the data directory.
    
    Args:
        data_dir: Path to the directory containing PDF files
        
    Returns:
        List of loaded documents
    """
    documents = []
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")
    
    print(f"Found {len(pdf_files)} PDF file(s). Loading...")
    
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        documents.extend(docs)
    
    print(f"Successfully loaded {len(documents)} pages from {len(pdf_files)} PDF(s)\n")
    return documents


def split_documents_by_date(documents: list) -> list:
    """
    Split documents into chunks based on date entries (format: DD/MM/YYYY).
    This preserves diary entries as logical units.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of split documents with date metadata
    """
    split_docs = []
    date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
    
    for doc in documents:
        content = doc.page_content
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 0)
        
        # Find all dates in the document
        dates = re.finditer(date_pattern, content)
        date_positions = [(m.start(), m.group()) for m in dates]
        
        if not date_positions:
            # No dates found, treat whole page as one chunk
            split_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'source': source,
                        'page': page,
                        'date': 'Unknown'
                    }
                )
            )
            continue
        
        # Split content by date entries
        for i, (pos, date) in enumerate(date_positions):
            # Get content from current date to next date (or end of document)
            start_pos = pos
            if i < len(date_positions) - 1:
                end_pos = date_positions[i + 1][0]
            else:
                end_pos = len(content)
            
            entry_content = content[start_pos:end_pos].strip()
            
            if entry_content:
                split_docs.append(
                    Document(
                        page_content=entry_content,
                        metadata={
                            'source': source,
                            'page': page,
                            'date': date
                        }
                    )
                )
    
    print(f"Split documents into {len(split_docs)} date-based entries\n")
    return split_docs
