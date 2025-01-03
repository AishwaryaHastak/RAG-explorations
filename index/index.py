# index.py

import argparse
import glob
import torch
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS  
from langchain.retrievers import BM25Retriever

class Index:
    def __init__(self, 
        model_name = "sentence-transformers/all-mpnet-base-v2",
        chunk_size = 1000, 
        chunk_overlap = 100,
        k = 4,
        vector_db_name = "vector_db",
        keyword_db_name = "keyword_db"
        ): 
        """
        Initialize the Index class with the given parameters.

        :param model_name: The name of the pre-trained model to use for embeddings
        :type model_name: str
        :param chunk_size: The size of the chunks to split the text into
        :type chunk_size: int
        :param chunk_overlap: The overlap between the chunks
        :type chunk_overlap: int
        :param k: The number of documents to return from keyword search
        :type k: int
        :param vector_db_name: The name of the vector database to save the index to
        :type vector_db_name: str
        :param keyword_db_name: The name of the keyword database to save the index to
        :type keyword_db_name: str
        """
        
        # Initialize text splitter parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Number of documents to return from keyword search
        self.k = k

        # Initialize embedding model
        model_name = model_name
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        self.vector_db_name = vector_db_name
        self.keyword_db_name = keyword_db_name

    def read_pdfs(self):
        """
        Reads all PDF files from the ./data directory, loads them using PyPDFLoader,
        and returns a list of loaded documents. Prints debug information showing the
        number of files being read and the number of documents loaded.

        :return: List of loaded documents
        """
        pdf_files = glob.glob("../data/*")
        # List to store all loaded documents
        documents = []
        print(f'[DEBUG] Reading {len(pdf_files)} files') 
        # Loop through all matching PDF files and load them
        for file_path in pdf_files:
            print(file_path)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())  # Load each PDF and add to the list

        print(f'[DEBUG] Loaded {len(documents)} files') 
        return documents

    def create_index(self): 
        # Load PDF files
        """
        Creates a FAISS vector database and a keyword database from all PDF files
        in the ./data directory.

        The function first loads all PDF files using PyPDFLoader, then applies the
        RecursiveCharacterTextSplitter to split each document into chunks of the
        specified size with the specified overlap. The chunks are then embedded
        using the specified HuggingFace model and the FAISS vector database is
        created and saved to the specified folder. The saved vector database can
        then be loaded for querying later.

        :return: The created FAISS vector database and keyword database
        """
        documents = self.read_pdfs()
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        # Apply the splitter to documents
        all_splits = text_splitter.split_documents(documents)
        print(f'[DEBUG] Creating index with {len(all_splits)} chunks...')
        
        # # Create FAISS vector database       
        vector_db = FAISS.from_documents(all_splits, self.embeddings)   
        vector_db.save_local(folder_path = self.vector_db_name, index_name = self.vector_db_name) 
        print(f'[DEBUG] Vector database saved to {self.vector_db_name}')

        # Create keyword database
        keyword_db = BM25Retriever.from_documents(all_splits, k=self.k)
        with open(f'{self.keyword_db_name}.pkl', 'wb') as f:
            pickle.dump(keyword_db, f)
        print(f'[DEBUG] Keyword database saved to {self.keyword_db_name}')
        
        return vector_db, keyword_db
    
    def load_index(self): 
        # Load vector database
        """
        Loads a previously saved FAISS vector database and keyword database from the specified
        folder paths.

        The method attempts to load both the vector database and the keyword database using
        the folder and index names specified by `vector_db_name` and `keyword_db_name`
        attributes respectively. If loading fails for any reason, an error message is 
        printed with the exception details.

        :return: A tuple containing the loaded vector database and keyword database.
        :rtype: tuple
        """
        try:
            vector_db = FAISS.load_local(folder_path = self.vector_db_name, index_name = self.vector_db_name, embeddings = self.embeddings)
            print(f'[DEBUG] Vector database loaded from {self.vector_db_name}')
        except Exception as e:
            print(f'[DEBUG] Vector database could not be loaded from {self.vector_db_name} due to the following error: {str(e)}')
        
        # Load keyword database
        try:
            with open(f"{self.keyword_db_name}.pkl", 'rb') as file_path:
                keyword_db = pickle.load(file_path) 
            print(f'[DEBUG] Keyword database loaded from {self.keyword_db_name}')
        except Exception as e:
            print(f'[DEBUG] Keyword database could not be loaded from {self.keyword_db_name} due to the following error: {str(e)}')

        return vector_db, keyword_db
    

def main():
    # Initialize argparse for command-line argument parsing
    parser = argparse.ArgumentParser(description="Create and load FAISS index for PDF files.")
    
    # Define arguments with default values
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", 
                        help="The model to use for generating embeddings (default: sentence-transformers/all-mpnet-base-v2).")
    parser.add_argument("--chunk_size", type=int, default=1000, 
                        help="Size of chunks for splitting text (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=100, 
                        help="The overlap between chunks (default: 100).")
    parser.add_argument("--k", type=int, default=4, 
                        help="Number of documents to return for keyword retriever (default: 4).")
    parser.add_argument("--vector_db_name", type=str, default="vector_db", 
                        help="The name for saving the vector database (default: vector_db).")
    parser.add_argument("--keyword_db_name", type=str, default="keyword_db", 
                        help="The name for saving the keyword database (default: keyword_db).")
    parser.add_argument("--load", action="store_true", 
                        help="If set, it will attempt to load an existing vector database instead of creating a new one.")
    
    # Parse arguments
    args = parser.parse_args()

    # Print the default values and allow user to modify
    print(f"Model name (default: {args.model_name}): ", end="")
    model_name = input() or args.model_name
    print(f"Chunk size (default: {args.chunk_size}): ", end="")
    chunk_size = input() or args.chunk_size
    print(f"Chunk overlap (default: {args.chunk_overlap}): ", end="")
    chunk_overlap = input() or args.chunk_overlap
    print(f"Vector DB name (default: {args.vector_db_name}): ", end="")
    vector_db_name = input() or args.vector_db_name
    print(f"Keyword DB name (default: {args.keyword_db_name}): ", end="")
    keyword_db_name = input() or args.keyword_db_name
    print(f"Number of documents to return for keyword retriever (default:{args.k}): ", end="")
    k = input() or args.k
    print(f"Load existing database? (default: {args.load}): ", end="")
    load = input() or args.load

    # Create Index object with parsed parameters
    index = Index(model_name=model_name, chunk_size=int(chunk_size), 
                  chunk_overlap=int(chunk_overlap), vector_db_name=vector_db_name,
                  keyword_db_name = keyword_db_name, k = k)

    # Conditionally either create a new index or load an existing one
    if args.load:
        print("Loading existing vector database...")
        vector_db = index.load_index()
        if vector_db:
            print("Vector database loaded successfully!")
        else:
            print("Failed to load vector database.")
    else:
        print("Creating a new index...")
        index.create_index()
        print("New vector database created successfully!")

if __name__ == "__main__":
    main()