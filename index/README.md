
# FAISS Vector Database Creation and Loading

This script allows you to create a new vector database or load an existing one using the FAISS library. You can configure the parameters for the database, such as the model used for embeddings, chunk size, chunk overlap, and the name of the vector database.

## Prerequisites

Before running the script, ensure that you have the following dependencies installed:

- Python 3.x
- Required libraries:
  - `langchain`
  - `faiss`
  - `torch`
  - `sentence-transformers`

You can install these dependencies using `pip`:

```
pip install langchain faiss-cpu torch sentence-transformers
```

## Usage

### 1. Create a New Vector Database
To create a new vector database, run the following command. You can specify the model to use, the chunk size for splitting documents, the overlap between chunks, and the name for the vector database.

```
python index.py --model_name "sentence-transformers/your-model" --chunk_size 500 --chunk_overlap 50 --vector_db_name "new_vector_db"
```
 
### Parameters: 
| Parameter                | Description                                                                                       | Default Value                  |
|--------------------------|---------------------------------------------------------------------------------------------------|--------------------------------|
| `--model_name`            | The model to use for generating embeddings.                                            | `sentence-transformers/all-mpnet-base-v2` |
| `--chunk_size`            | The size of chunks for splitting the documents.                                                   | `1000`                         |
| `--chunk_overlap`         | The overlap between chunks.                                                                       | `100`                          |
| `--vector_db_name`        | The name of the directory where the vector database will be saved.                                | `vector_db`                    |




### 2. Load an Existing Vector Database
If you already have a saved vector database and you want to load it, use the following command:

```
python index.py --load --vector_db_name "existing_vector_db"
```
#### Parameters:

| Parameter              | Description                                                                                       | Default Value                  |
|------------------------|---------------------------------------------------------------------------------------------------|--------------------------------|
| `--load`               | Flag to indicate that you want to load an existing vector database.                               | ` ` (no default value)         |
| `--vector_db_name`     | The name of the existing vector database you want to load.                                        | `vector_db`                    |

 