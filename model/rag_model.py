import os
from collections import deque
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from operator import itemgetter

""" GLOBALS """
# Directories
base_dir = os.path.dirname(os.path.abspath(__file__))

# RAG components
rag_chain = None                                # RAG model chain
MAX_HISTORY_LENGTH = 5                          # Max number of exchanges to keep in history
chat_history = deque(maxlen=MAX_HISTORY_LENGTH) # Stores user message/question and bot responses

# Directory for persistent vector store
DB_DIR = os.path.join(base_dir, "chroma_db")
os.makedirs(name=DB_DIR, exist_ok=True)

# Training data
PDF_FILE = "Disaster_Preparedness_First_Aid_Handbook_Plaintext.pdf"

# Initialize RAG model
def initialize_RAG():
    global rag_chain
    if rag_chain is not None:
        return rag_chain
    print("Initializing RAG model.")    # Debug

    # Load PDF document
    pdf_path = os.path.join(base_dir, "pdf", PDF_FILE)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,                            # Default is 1000, increased for better context and for testing
        chunk_overlap=200,                          # Default is 200
        separators=["\n\n", "\n", ".", "?", "!"]    # Sentence boundaries
    )
    chunks = text_splitter.split_documents(docs)

    # Embeddings and vector store
    embeds = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Persistent vector store
    vector_store = None
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        print("Creating persistent vector store to disk.")    # Debug
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeds,
            persist_directory=DB_DIR
        )
    else:
        print ("Loading existing persistent vector store from disk.")    # Debug
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeds
        )

    # Generation
    llm = OllamaLLM(
        model="disaster-alert-bot:latest",  # Small model due to limited resources and for testing
    )

    # Prompt template (new to langchain since RetrievalQA is deprecated)
    template = """
You are a disaster preparedness assistant. 
Answer only using the context provided below. 
If the answer cannot be found in the context, say clearly:
"Sorry, I don’t have enough information from my sources to answer that."

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    

    # RAG Components for LCEL Chain
    rag_chain = (
        {
            "question": itemgetter("question"),
            "chat_history": lambda _: "",
            "context": lambda x: "\n\n".join(
                [d.page_content for d in vector_store.similarity_search(x["question"], k=1)]    # Retriever moved and simplified to here
            )
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG model initialized.")    # Debug
    return rag_chain

# Stream response from RAG model
def stream_response_RAG(question: str) -> None:
    chain = initialize_RAG()
    print("Bot: ", end="", flush=True)

    response_chunks = []
    for chunk in chain.stream({"question" : question}):
        print(chunk, end="", flush=True)
        response_chunks.append(chunk)               # Collect chunks
    response_text = "".join(response_chunks)        # Combine chunks into full response
    chat_history.append((question, response_text))  # Store and update chat history
    print("\n")

# Debug from terminal
if __name__ == "__main__":
    print("DisasterAlertBot RAG Terminal Interface")
    print("Type your question below (type 'exit' or 'quit' to quit):\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting DisasterAlertBot. Stay safe!")
            break
        
        try:
            stream_response_RAG(question=user_input)
        except Exception as e:
            print(f"Error: {e}\n")