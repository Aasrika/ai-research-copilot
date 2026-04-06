# AI Research Copilot

A multi-agent RAG system for analyzing research papers.

## Features
- PDF parsing and chunking
- FAISS vector search
- LLM-based Q&A with citations
- Groq-powered fast inference

## Tech Stack
- LangChain
- FAISS
- SentenceTransformers
- Groq API

## Status
Phase 1 Complete 

## 🚀 Phase 2: Grounded RAG System

- Implemented Retrieval-Augmented Generation (RAG) pipeline
- Integrated Groq LLM (LLaMA 3.3 70B)
- Added citation-based answering ([Source N])
- Enabled paper-specific filtering
- Improved retrieval using MMR
- Built Streamlit UI for querying research papers
- Added source traceability (page, section)

### Features
- Ask questions over research papers
- View supporting source chunks
- Compare papers (basic)
- Prevent hallucinations via grounded prompting