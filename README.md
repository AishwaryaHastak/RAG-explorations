# RAG Explorations: RAG Techniques with FAISS and LangChain

This repository demonstrates the implementation of different **Retrieval-Augmented Generation (RAG)** techniques using **LangChain**, vector stores, and LLM models, and evaluating them using **Langsmith**. The project is organized into different notebooks, each focusing on a specific technqiue of implementing a RAG pipeline.

## Project Structure

- **`index.py`**: Handles the creation and loading of FAISS vector databases. [View the README for `index.py`](./index/README.md)
- **`notebooks/`**: Collection of Jupyter notebooks demonstrating different RAG techniques.
    - **`rag_technique_1.ipynb`**: Demonstrates a basic RAG method using FAISS for document retrieval and a transformer model for generation.
    - **`rag_technique_2.ipynb`**: Another advanced RAG method, possibly involving fine-tuning or hybrid retrieval models.
- **`data/`**: Contains the sample data (like PDFs) that are processed by `index.py` to create a vector database.
- **`vector_db`**: FAISS vector database
