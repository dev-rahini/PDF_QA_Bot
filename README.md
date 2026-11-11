# PDF Document QA & Semantic RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) system built in Python using LangChain, Hugging Face embeddings, and FAISS vector indexing to query unstructured PDF documents.

## Pipeline Architecture
1. **Document Ingestion:** Parses multi-page PDF documents using `PyPDFLoader`.
2. **Text Chunking:** Applies `RecursiveCharacterTextSplitter` (chunk size: 500 characters, overlap: 50 characters) to preserve contextual boundaries.
3. **Dense Vector Representation:** Generates 384-dimensional dense semantic embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2`.
4. **Vector Database & Indexing:** Stores embeddings locally in a `FAISS` (Facebook AI Similarity Search) index for high-throughput similarity search.
5. **Similarity Retrieval:** Retrieves top-k semantically relevant context chunks for a natural language query and formats a grounded prompt to prevent hallucinations.

## Tech Stack
- **Language:** Python
- **Orchestration:** LangChain
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS (CPU)
- **Document Parser:** PyPDFgit init