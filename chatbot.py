import os
from dotenv import load_dotenv 
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory 
from langchain.globals import set_verbose, set_debug 
from chromadb.config import Settings as ChromaSettings 
import warnings
warnings.filterwarnings("ignore")

print("Starting chatbot... Wait a moment please.")

# Load environment variables and set parameters to avoid logging
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false" # Deactivate tracing LangSmith
os.environ["LANGCHAIN_API_KEY"] = "" 

set_verbose(False)
set_debug(False)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


# File configuration
file_paths = [
     "./Nikola_Tesla.pdf"
]
CHROMA_DB_PATH = "./chroma_db_tesla" 

# Determine the device for embeddings
if torch.cuda.is_available():
    EMBEDDING_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    EMBEDDING_DEVICE = "mps"
else:
    EMBEDDING_DEVICE = "cpu"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': EMBEDDING_DEVICE}
)

# Load or create vector store
chroma_client_settings = ChromaSettings(
    anonymized_telemetry=False, 
    is_persistent=True, 
    persist_directory=CHROMA_DB_PATH 
)

if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings, client_settings=chroma_client_settings)
else:
    docs = []
    try:
        loader = PyPDFLoader(file_paths[0]) 
        docs = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        print("Please ensure the file paths are correct and the documents are accessible.")
        exit() 

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(docs)
    texts = filter_complex_metadata(texts) # Remove complex metadata if any

    db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)

# Initialize LLM and conversational chain
llm = OllamaLLM(model="llama3") # Or "llama3:8b-instruct-q4_K_M" for faster inference

# Initialize conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the retrievalQA chain with memory
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 5}), 
    memory=memory, 
    return_source_documents=False 
)
print("Everything set! Type 'exit' to quit.")

# Interactive chat loop
while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: Thinking...")
        result = chain.invoke({"query": user_input})
        print(f"Chatbot: {result['result']}")

    except KeyboardInterrupt: 
        print("\nChatbot: Interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"Chatbot Error: An unexpected error occurred: {e}")
        print("Please try again or type 'exit' to quit.")
