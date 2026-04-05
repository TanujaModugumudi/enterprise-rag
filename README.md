# Enterprise Document Intelligence System

A production-grade RAG (Retrieval-Augmented Generation) pipeline for intelligent 
document analysis across HR, Legal, and Finance domains.

## Features
- Smart Q&A with citations (hybrid search + reranking)
- Structured JSON extraction from documents
- Multi-document comparison
- Risk and insight detection
- RAGAS evaluation dashboard

## Tech Stack
- **LLM**: Groq API (Llama 3)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Retrieval**: LangChain EnsembleRetriever (BM25 + semantic)
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Evaluation**: RAGAS
- **Backend**: FastAPI
- **UI**: Gradio

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure
```
enterprise-rag/
├── data/raw/          # Source PDFs (HR, Legal, Finance)
├── src/
│   ├── ingestion/     # PDF loading, chunking, embedding
│   ├── retrieval/     # Hybrid search + reranking
│   ├── generation/    # LLM + citation engine
│   ├── extraction/    # Structured JSON extraction
│   └── evaluation/    # RAGAS metrics
├── config/            # Settings and constants
└── main.py            # Application entry point
```