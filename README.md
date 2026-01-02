# Diary-based RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your journal PDFs using LangChain and OpenAI.

## Features

- **PDF Loading**: Automatically loads all PDF documents from the `data/` folder
- **Document Chunking**: Splits documents into manageable chunks for better retrieval
- **Vector Store**: Uses ChromaDB for efficient semantic search
- **RAG Chain**: Combines retrieval with generative AI for accurate answers
- **Interactive Mode**: Ask questions in real-time
- **Batch Mode**: Process multiple questions at once
- **Source Attribution**: Shows which diary entries were used to answer questions

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Create a `.env` file in the project root (use `.env.example` as a template):

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

### 3. Add Your Diary PDFs

Place your PDF files in the `data/` folder. The system will automatically discover and process them.

## Usage

### Interactive Mode

Run the main program for interactive question answering:

```bash
python main.py
```

Then ask questions like:
- "What happened on my birthday?"
- "What are my main concerns?"
- "Summarize the key events from my diaries"

Type `exit` to quit.

### Batch Mode

Process multiple questions at once:

```bash
python batch_query.py
```

Edit the `questions` list in `batch_query.py` to ask your own questions.

### Custom Script

Import the modules in your own script:

```python
from document_loader import load_pdf_documents, split_documents
from vector_store import create_or_load_vector_store
from rag_chain import create_rag_chain, query_rag_chain

# Initialize
documents = load_pdf_documents()
split_docs = split_documents(documents)
vector_store = create_or_load_vector_store(split_docs)
qa_chain = create_rag_chain(vector_store)

# Query
result = query_rag_chain(qa_chain, "Your question here")
print(result["answer"])
```

## Project Structure

```
diary_based_rag_qa/
├── data/                    # Your diary PDF files
├── chroma_db/              # Vector store (auto-created)
├── main.py                 # Interactive QA interface
├── batch_query.py          # Batch processing script
├── document_loader.py      # PDF loading and splitting
├── vector_store.py         # Vector database management
├── rag_chain.py           # RAG chain and querying
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # API keys (create from .env.example)
└── README_SETUP.md        # This file
```

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `LLM_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)

## How It Works

1. **Document Loading**: PDFs are loaded and converted to text
2. **Chunking**: Documents are split into overlapping chunks
3. **Embedding**: Text chunks are converted to vector embeddings
4. **Storage**: Embeddings are stored in ChromaDB for fast retrieval
5. **Retrieval**: When you ask a question, similar chunks are retrieved
6. **Generation**: The LLM uses retrieved context to generate an answer

## Performance Tips

- First run will take longer as it processes and embeds all documents
- Subsequent runs will load from cache (much faster)
- To reset and re-process all documents, delete the `chroma_db/` folder

## Troubleshooting

### "No PDF files found"
- Ensure your PDF files are in the `data/` folder
- Check file extensions are `.pdf` (case-sensitive)

### "OpenAI API key not found"
- Create a `.env` file with `OPENAI_API_KEY=your_key`
- Ensure your API key is valid at https://platform.openai.com/api-keys

### Out of memory errors
- Reduce `CHUNK_SIZE` in `config.py`
- Process fewer PDFs at a time

## API Costs

This system uses OpenAI's API which incurs costs:
- Embedding API: ~$0.02 per 1M tokens
- Chat completion API: ~$0.5-$2 per 1M tokens (varies by model)

Monitor your usage at https://platform.openai.com/account/usage/overview
