# RAG Explorations: RAG Techniques with FAISS and LangChain

This repository demonstrates the implementation of different **Retrieval-Augmented Generation (RAG)** techniques using **LangChain**, vector stores, and LLM models, and evaluating them using **Langsmith**. The project is organized into different notebooks, each focusing on a specific technqiue of implementing a RAG pipeline.

## Project Structure

- **`index/`**: Handles the creation and loading of FAISS vector databases. [View the README for `index.py`](./index/README.md)
- **`notebooks/`**: Collection of Jupyter notebooks demonstrating different RAG techniques. 
- **`data/`**: Contains the sample data (like PDFs) that are processed by `index.py` to create a vector database.
- **`vector_db`**: FAISS vector database
