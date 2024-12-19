# app1.py
# app.py
"""
RAG Chatbot
This application provides real-time support by leveraging LLM technology to answer document-related queries.

Main Components:
- Frontend: Streamlit interface 
- Search: FAISS vector similarity search
- Models: Llama 3.2 3B for response generation
"""

# Core dependencies
import torch
from torch import cuda
import gc
import os
import warnings
import pickle

# ML/NLP dependencies
import transformers
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter  

# AWS dependencies
import boto3
from botocore.exceptions import NoCredentialsError

# UI dependencies
import streamlit as st
from time import time

# Intel Optimization
from optimum.intel import OVModelForCausalLM
from optimum.exporters.openvino import export_from_model
 
# Application Configuration
st.set_page_config(
    page_title="DocuBot",
    page_icon="📚",   
    layout="wide"
)

# Performance Optimization Settings 
os.environ["OMP_NUM_THREADS"] = "16"  # CPU thread optimization
os.environ["MKL_NUM_THREADS"] = "16"  # Math kernel optimization
torch.set_num_threads(16)             # PyTorch threading

# Model Configuration
MODEL_ID = 'meta-llama/Llama-3.2-3B-Instruct'
MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct' 
ACCESS_TOKEN = ""
BUCKET_NAME = 'carnival-techassistant'
 
embeddings = None
# Embeddings and vector database setup
model_name = "sentence-transformers/all-mpnet-base-v2"  
model_kwargs = {"device": "cuda" if cuda.is_available() else "cpu"}  # Use GPU if available
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs) 
vectordb = FAISS.load_local(folder_path='faiss_db_test',index_name="faiss_db_test",embeddings=embeddings)

# Load the FAISS vector store (retriever_vectordb)
retriever_vectordb = vectordb.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs={"k": 5, 
                   "score_threshold": 0.55})

# Load BM25 retriever from disk
def load_bm25_retriever(file_path):
    with open(file_path, 'rb') as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever
keyword_retriever = load_bm25_retriever("bm25_retriever.pkl") 
keyword_retriever.k =  2
ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],
                                       weights=[0.7, 0.3])

# Prompt template setup (does not ask for the model once it's been provided)
prompt_template_response = PromptTemplate(
    input_variables=["question", "context"],
    template=""" 
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an intelligent assistant for Carnival Cruise technicians, designed to help with equipment-related questions for operational support and guidance.
    Provide a detailed, complete and holistic answer to the question using the information from the given context. 
    If the question is not specific and more details are needed to answer accurately, then DO NOT provide a general answer and always ASK for more details.
    Include any important warnings from the context in your response. Ensure your response is well-formatted and easy to understand.
    Do NOT provide general information or assumptions outside the given context. 
    If the question is about equipment troubleshooting problems, then add at the end of the answer: "In case the information did not fit your needs, please try:\n- Reformulate your question \n- Refer to the manual \n- Contact your superior or an specialist."
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}
    Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
) 
# If the question is not specific or the context is too extensive, do not make assumptions and always request more specific details to narrow down the answer.
    
 
# Define a separate template for model-specific questions (when model info is needed)
prompt_template_ask_model = PromptTemplate(
    input_variables=["question", "context"],
    template=""" 
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an intelligent assistant for Carnival Cruise technicians, designed to help with equipment-related questions for operational support and guidance.
    DO NOT extract and DO NOT assume the model name from the context. The model name MUST always only be extracted from the QUESTION.
    If the model name is NOT provided in the question, then NEVER provide a general answer and ALWAYS ASK for the model by responding with EXACTLY ONLY: "To assist you, I need the exact model of the [equipment]. What is the specific model number or name?"
    Only if a model name or maker name is specified in the question, you may then use the relevant information in the Context to answer the question. 
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: {context}
    QUESTION: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
) 


# Prompt Template for query rewriting
question_template = PromptTemplate(
    input_variables=['chat_history', 'question'],
    template=""" 
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Given the chat history and the human's follow up question, do not attempt to answer the question and only create a new question that DOES NOT change or alter the words in the follow up question,use context from previous human questions if relevent, and ALWAYS ADD the EXACT model name and EXACT equipment mentioned in the chat history to the new question. 
    Return ONLY the new question with NO extra text or explanations.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Chat History: {chat_history}
    Follow Up Question: {question} 
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Question:
    """
    ) 
# use context from previous human questions only if relevent,
# Given the chat history and the human's follow up question, create a new question that DOES NOT change or alter the words in the latest human question, use context from previous human questions in the chat history if relevent to formulate the new question.

#    DO NOT attemppt to answer the question. ONLY follow the instructions above to CREATE a new question.
     
# The new question should be closely related to the follow up question.
# The new question MUST be formulated from the follow up question, using context from previous human questions only if relevent to create the new question.
     
#  Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the latest human question, use context from previous human questions in the chat history if relevent, and ALWAYS ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
   
    # Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the latest human question, use the chat history if needed, and ALWAYS ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
    
# Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the latest human question, use the chat history if needed, and ALWAYS ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
    
    # Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the latest human question but ALWAYS ONLY ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
    
    # Given the chat history and the follow up question, create a new question that does not change or alter the words in the latest human question but ALWAYS ONLY ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
   
# Given the chat history and the human's follow-up question, create a new question always using the follow-up question if it is a question. If not a question then use the most recent question from the chat history.
#     The new question should NOT change or alter the words in the question but ALWAYS ONLY ADD the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
#     Return ONLY the modified question, with no extra text or explanation.
    
    #  Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the follow up question (if its a question or else the most recent question) but ALWAYS ONLY ADDS the EXACT model name and the EXACT equipment type mentioned in the chat history to it.
    # Return ONLY the new question with NO extra text or explanations.
     
    # Given the chat history and the human's follow up question, create a new question that does not change or alter the words in the most recent human question but only adds the exact model name and the exact equipment type mentioned in the chat history to it.
 
# Have the title be always visible on the screen even on scrolling
def stick_it_good(): 
    # make header sticky.
    st.markdown(
        """
            <div class='fixed-header'/>
            <style>
                div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                    position: sticky;
                    top: 2.875rem;
                    background-color: white;
                    z-index: 999;
                }
            </style>
        """,
        unsafe_allow_html=True
    )

# Initialize S3 client
s3_client = boto3.client('s3')

def get_presigned_url(bucket_name, file_name, page_number):
    """
    Generate a pre-signed URL for accessing documents in S3.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        file_name (str): Path to the file in S3
        page_number (int): PDF page number to link to
    
    Returns:
        str: Pre-signed URL with page reference
    """
    try:
        print(f"Generating presigned URL for Bucket: {bucket_name}, Key: {file_name}")
        response = s3_client.generate_presigned_url('get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600)  # URL expires in 1 hour
        return f"{response}#page={page_number}"
    except NoCredentialsError:
        st.error("Credentials not available.")
        return None

# Global variables for model and tokenizer
model = None
tokenizer = None  
question_gen_model = None 
# reranker = None

@st.cache_resource(show_spinner=False) 
def load_model(): 
    """
    Load and cache the primary LLM model with OpenVINO optimization.
    
    Returns:
        HuggingFacePipeline: Configured model pipeline for main response generation
    
    Note:
        Stores base model in session state for reuse in question generation
    """ 
    # Check if model already exists in cache
    if not os.path.exists("model_cache"):
        try:
            model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)
            model.save_pretrained("model_cache") 
            export_from_model(model, output="model_cache", task="text-generation-with-past", library="transformers")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        model = OVModelForCausalLM.from_pretrained("model_cache") 
    
    # Save the base LLM model 
    st.session_state.base_model = model 

    # return model
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu' 
     
    # Load tokenizer and model pipeline
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, 
        token=ACCESS_TOKEN,  
        device=device)  
    st.session_state.tokenizer = tokenizer

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=1024, 
        top_k=5, 
        temperature=0.2, 
        token=ACCESS_TOKEN,
        eos_token_id=tokenizer.eos_token_id,  
        early_stopping=False,  # Disable early stopping to allow complete responses
        device=0 if cuda.is_available() else -1,
        )  

    st.session_state.model = HuggingFacePipeline(pipeline=query_pipeline)
    return st.session_state.model

@st.cache_resource(show_spinner=False) 
def load_question_gen_model(): 
    """
    Load question generation model using the cached base model.
    
    Returns:
        HuggingFacePipeline: Configured pipeline for question reformulation
    
    Note:
        Reuses base model from session state to avoid duplicate loading
    """ 
      
    # Check if model already exists in cache
    if not os.path.exists("model_cache"):
        model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True)
        model.save_pretrained("model_cache") 
        export_from_model(model, output="model_cache", task="text-generation-with-past", library="transformers")
    else:
        model = OVModelForCausalLM.from_pretrained("model_cache")  

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model, 
        tokenizer=st.session_state.tokenizer,
        max_new_tokens=50,
        # repetition_penalty=1.2,
        top_k=5, 
        temperature=0.1, 
        token=ACCESS_TOKEN,
        eos_token_id=st.session_state.tokenizer.eos_token_id,  
        device=0 if cuda.is_available() else -1,
        )

    st.session_state.question_gen_model = HuggingFacePipeline(pipeline=query_pipeline) 
    return st.session_state.question_gen_model
    
llm = load_model()
question_llm = load_question_gen_model()

def response_generator(response):
    """
    Generate words from response string for streaming output.
    
    Args:
        response (str): Complete response text
        
    Yields:
        str: Individual words with spacing
    """
    for word in response.split():
        yield word + " " 

# Function to find related documents
def doc_to_find(docs):
    """
    Extract page numbers and source information from documents.
    
    Args:
        docs (list): List of document objects from vector store
        
    Returns:
        list: Tuples of (page_number, source_file)
    """
    return [(doc.to_json()['kwargs']['metadata']['page'], doc.to_json()['kwargs']['metadata']['source']) for doc in docs]

# Function to get LLM response with caching
@st.cache_data(show_spinner=False)
def get_llm_response(question, chat_history):  
    """
    Generate LLM response with caching for improved performance.
    
    Args:
        question (str): User query
        chat_history (list): Previous conversation context
        
    Returns:
        dict: Contains 'answer' and 'source_documents'
    """

    # Check if initial run
    prompt_template = prompt_template_ask_model if st.session_state.first_run else prompt_template_response
    
    # Conversational retrieval chain setup
    qa_1 = ConversationalRetrievalChain.from_llm(   
        llm=llm,  
        # retriever=vectordb.as_retriever(
        #     search_type = "similarity_score_threshold", 
        #     search_kwargs={"k": 1 if st.session_state.first_run else 4,
        #                    'score_threshold': 0.4}),
        retriever = ensemble_retriever, 
        condense_question_prompt = question_template, 
        condense_question_llm = question_llm, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt": prompt_template},    
        verbose = True
        ) 
    chain = qa_1({"question": question, 'chat_history': chat_history})

    return chain 

# Load model and tokenizer only once
if 'model' not in st.session_state:
    with st.spinner("Loading the model..."):
        load_model()

warnings.filterwarnings("ignore")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [] 
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
# Initialize session state
if "first_run" not in st.session_state:
    st.session_state.first_run = True 

with st.container():
    st.title("Digital Tech Assistant for Carnival")
    stick_it_good()
    
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar with styled buttons
with st.sidebar:
    # Reset Chat button with icon
    if st.button("🔄 Reset Chat", key="reset", help="Reset the conversation"):
        st.session_state.messages = []      # Clear the chat messages
        st.session_state.chat_history = []  # Clear the chat history
        st.session_state.first_run = True   # Set first_run back to True for a fresh start
        st.cache_data.clear()  # Clear the cached data
        st.rerun()

    # # Clear Cache button with icon
    # if st.button("🧹 Clear Cache", key="clear", help="Clear cached chat history"):
        # st.cache_data.clear()  # Clear the cached data

question = st.chat_input("Type your question here...")

if question:
    # Capture start time
    start_time = time()
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question}) 

    with st.chat_message("user"):
        st.markdown(question)

    # Prepare chat history for the model  
    # if len(st.session_state.chat_history) > 4:
    # # Keep only the 3 past conversations to reduce response latency
    #     chat_history = st.session_state.chat_history[:2] + st.session_state.chat_history[-2:]
    if len(st.session_state.chat_history) > 3:
    # Keep only the 3 past conversations to reduce response latency
        chat_history = st.session_state.chat_history[:2] + st.session_state.chat_history[-1:]
    else:
        chat_history = st.session_state.chat_history

    with st.spinner("Generating response..."): 
        chain = get_llm_response(question,chat_history)

    response_llama = chain['answer'] 
    source_documents = chain['source_documents']

    # If no context was fetched, ask to reformulate the question
    if len(source_documents) == 0 and st.session_state.first_run == False:
        print('len(source_documents)',len(source_documents))
        response_llama = "No relevant information was found, please reformulate your question."
    elif len(source_documents) > 4 and st.session_state.first_run == False: 
        source_documents = source_documents[:4] 

    # If the llm response asks for model clarification, then do not show related documents
    if st.session_state.first_run == True: source_documents = []  
    
    st.session_state.first_run = False  
    stop_time = time() 
    # Print the elapsed time for the processing cycle
    print(f"Cycle took {stop_time - start_time:.2f} seconds.") 

    st.subheader("Assistant's Response")    
    st.markdown(response_llama, unsafe_allow_html=True)   

    docs_info = list(set(doc_to_find(source_documents))) 
    with st.expander("  Related Documents"):
        for page, source in docs_info: 
            file_name = 'data/' + rf'{source}'.split("\\")[1]
            doc_name = source.split('.')[2].strip()
            presigned_url = get_presigned_url(BUCKET_NAME, file_name, page+1)   # PyPDF pages start at 0, so +1 to match PDF's actual page numbering

            col1, col2 = st.columns([5, 1]) 
            with col1:
                st.write(f"Page: {page+1}, Document: {doc_name}")

            with col2:
                if presigned_url:
                    download_link = f"[Link]({presigned_url})"
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.write("Link not available") 
        
    st.session_state.chat_history.append((question, response_llama)) 
    # Add assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_llama}) 
    gc.collect()