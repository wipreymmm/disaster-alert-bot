import os
import time
from collections import deque
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from services.data_scrape import fetch_disaster_data
from typing import List

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = "Disaster_Preparedness_First_Aid_Handbook_Plaintext.pdf"
PDF_PATH = os.path.join(BASE_DIR, "pdf", PDF_FILE)
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

MAX_HISTORY_LENGTH = 5
chat_history = deque(maxlen=MAX_HISTORY_LENGTH)

TOP_K_CHUNKS = 3  # Increased to 3 since we have more diverse sources now

# -------------------- LOAD AND PREPARE DATA --------------------
def load_and_prepare_documents() -> List[Document]:
    """
    Load documents from multiple sources: web scraping (priority) + PDF (fallback).
    Returns a list of all Document chunks ready for vector store.
    """
    all_chunks = []
    
    # 1. Load PDF and split into chunks (fallback source)
    print("[DEBUG] Loading PDF...")
    start_time = time.time()
    try:
        loader = PyMuPDFLoader(PDF_PATH)
        pdf_docs = loader.load()
        print(f"[DEBUG] Loaded {len(pdf_docs)} PDF pages in {time.time() - start_time:.2f}s")
        
        # Split PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!"]
        )
        pdf_chunks = text_splitter.split_documents(pdf_docs)
        print(f"[DEBUG] Split PDF into {len(pdf_chunks)} chunks")
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {e}")
        pdf_chunks = []
    
    # 2. Fetch disaster data (web first, then PDF fallback)
    print("[DEBUG] Fetching disaster data from multiple sources...")
    fetched_docs = fetch_disaster_data(pdf_chunks=pdf_chunks)
    
    # 3. Split web-scraped documents into chunks (they might be large)
    print("[DEBUG] Splitting fetched documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!"]
    )
    
    for doc in fetched_docs:
        # Check if document is from web (not PDF) and needs splitting
        if doc.metadata.get("source", "").startswith("http"):
            # Web documents might be large, split them
            chunks = text_splitter.split_documents([doc])
            all_chunks.extend(chunks)
        else:
            # PDF chunks are already split, add as-is
            all_chunks.append(doc)
    
    print(f"[DEBUG] Total chunks prepared: {len(all_chunks)}")
    return all_chunks

# -------------------- INITIALIZE SYSTEM --------------------
print("[DEBUG] Initializing DisasterAlertBot with web + PDF sources...")
chunks = load_and_prepare_documents()

# -------------------- VECTOR STORE --------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("[DEBUG] Initializing vector store...")

start_time = time.time()
# Always recreate vector store to include fresh web data
# If you want to cache, add logic to check if web sources changed
vector_store = Chroma.from_documents(
    chunks, 
    embedding=embeddings, 
    persist_directory=DB_DIR
)
print(f"[DEBUG] Vector store created with {len(chunks)} chunks in {time.time() - start_time:.2f}s")

# -------------------- LLM --------------------
llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)
print("[DEBUG] LLM initialized")

# -------------------- PROMPT TEMPLATE --------------------
PROMPT_TEMPLATE = """
You are a disaster preparedness assistant with access to multiple authoritative sources.
Answer the question using the context provided. If the context contains partial information, answer as best as possible without inventing facts. 
If the context does not contain relevant information, respond: "Sorry, I don't have enough information from my sources to answer that."

When citing information, prioritize the most recent and authoritative sources.

Chat History:
{chat_history}

Context from multiple sources:
{context}

Question:
{question}

Answer:
"""

# -------------------- ASK FUNCTION --------------------
def ask_question(question: str) -> str:
    print(f"[DEBUG] Asking question: {question}")
    start_time = time.time()

    # Build chat history string
    history_text = ""
    for user_q, bot_a in list(chat_history)[-MAX_HISTORY_LENGTH:]:
        history_text += f"User: {user_q}\nBot: {bot_a}\n"
    print(f"[DEBUG] Chat history length: {len(chat_history)}")

    # Retrieve top-k chunks for context
    docs = vector_store.similarity_search(question, k=TOP_K_CHUNKS)
    
    # Build context with source attribution
    context_parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "Unknown source")
        content = d.page_content.strip()
        context_parts.append(f"[Source {i}: {source}]\n{content}")
    
    context = "\n\n".join(context_parts) or "No relevant context found."
    print(f"[DEBUG] Retrieved {len(docs)} relevant chunks in {time.time() - start_time:.2f}s")

    # Fill prompt
    prompt = PROMPT_TEMPLATE.format(
        chat_history=history_text,
        context=context,
        question=question
    )

    # Generate response from LLM
    gen_start = time.time()
    result = llm.generate([prompt])
    answer = result.generations[0][0].text.strip()
    print(f"[DEBUG] LLM response generated in {time.time() - gen_start:.2f}s")

    # Update chat history
    chat_history.append((question, answer))
    print(f"[DEBUG] Updated chat history; total exchanges: {len(chat_history)}")
    return answer

# -------------------- OPTIONAL: REFRESH WEB DATA --------------------
def refresh_web_data():
    """
    Refresh vector store with latest web data.
    Call this periodically or on-demand to update web sources.
    """
    global vector_store, chunks
    print("[DEBUG] Refreshing web data...")
    chunks = load_and_prepare_documents()
    vector_store = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("[DEBUG] Vector store refreshed with updated data")

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    print("DisasterAlertBot (RAG with Web + PDF sources)")
    print("Type your question below")
    print("Commands: 'exit' or 'quit' to exit | 'refresh' to update web sources\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Stay safe!")
            break
        elif user_input.lower() == "refresh":
            refresh_web_data()
            print("Web sources refreshed!\n")
            continue
        
        if not user_input:
            continue
            
        try:
            start = time.time()
            response = ask_question(user_input)
            print("Bot:", response)
            print(f"[DEBUG] Total time for this question: {time.time() - start:.2f}s\n")
        except Exception as e:
            print(f"Error: {e}\n")