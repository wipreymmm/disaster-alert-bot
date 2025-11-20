import os
import time
from collections import deque
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from .services.data_scrape import fetch_disaster_data

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = "Disaster_Preparedness_First_Aid_Handbook_Plaintext.pdf"
PDF_PATH = os.path.join(BASE_DIR, "pdf", PDF_FILE)
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

MAX_HISTORY_LENGTH = 5
chat_history = deque(maxlen=MAX_HISTORY_LENGTH)

TOP_K_CHUNKS = 5

# -------------------- LOAD AND PREPARE DATA --------------------
def load_pdf_chunks() -> List[Document]:
    try:
        loader = PyMuPDFLoader(PDF_PATH)
        pdf_docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!"]
        )
        pdf_chunks = text_splitter.split_documents(pdf_docs)
        for chunk in pdf_chunks:
            chunk.metadata["source_type"] = "pdf"
            chunk.metadata["priority"] = "high"
        return pdf_chunks
    except Exception as e:
        print(f"[ERROR] Failed to load PDF: {e}")
        return []

def load_web_chunks() -> List[Document]:
    try:
        web_docs = fetch_disaster_data(pdf_chunks=None)
        if not web_docs:
            return []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "?", "!"]
        )
        web_chunks = []
        for doc in web_docs:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["source_type"] = "web"
                chunk.metadata["priority"] = "medium"
            web_chunks.extend(chunks)
        return web_chunks
    except Exception as e:
        print(f"[ERROR] Failed to load web sources: {e}")
        return []

def initialize_vector_stores():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pdf_chunks = load_pdf_chunks()
    pdf_store = Chroma.from_documents(
        pdf_chunks,
        embedding=embeddings,
        persist_directory=os.path.join(DB_DIR, "pdf_store")
    )
    web_chunks = load_web_chunks()
    web_store = None
    if web_chunks:
        web_store = Chroma.from_documents(
            web_chunks,
            embedding=embeddings,
            persist_directory=os.path.join(DB_DIR, "web_store")
        )
    return pdf_store, web_store, embeddings

# -------------------- INITIALIZE --------------------
pdf_vector_store, web_vector_store, embeddings = initialize_vector_stores()
llm = OllamaLLM(model="llama3.2:3b", temperature=0.6)

# -------------------- PROMPT --------------------
PROMPT_TEMPLATE = """
You are DisasterAlertBot — an AI disaster preparedness assistant for the Philippines. 
...
Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""

# -------------------- RETRIEVAL --------------------
def retrieve_data(question: str, k: int = 5) -> tuple[List[Document], str]:
    pdf_docs = pdf_vector_store.similarity_search_with_score(question, k=k)
    combined_docs = []
    combined_docs.extend([doc for doc, _ in pdf_docs[:3]])
    
    if web_vector_store:
        web_docs = web_vector_store.similarity_search(question, k=2)
        combined_docs.extend(web_docs)
        return combined_docs, "pdf_and_web"
    return combined_docs, "pdf_only"

# -------------------- ASK FUNCTION --------------------
def ask_question(question: str) -> str:
    history_text = "".join(
        f"User: {q}\nBot: {a}\n" for q, a in list(chat_history)[-MAX_HISTORY_LENGTH:]
    )
    docs, _ = retrieve_data(question, k=TOP_K_CHUNKS)
    context_parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "PDF Handbook")
        source_type = d.metadata.get("source_type", "unknown")
        content = d.page_content.strip()
        label = f"PDF Handbook - Page {source}" if source_type == "pdf" else f"Web - {source}"
        context_parts.append(f"[Source {i}: {label}]\n{content}")
    context = "\n\n".join(context_parts) or "No relevant context found."
    prompt = PROMPT_TEMPLATE.format(chat_history=history_text, context=context, question=question)
    result = llm.generate([prompt])
    answer = result.generations[0][0].text.strip()
    chat_history.append((question, answer))
    return answer

# -------------------- REFRESH WEB DATA --------------------
def refresh_web_data():
    global web_vector_store
    web_chunks = load_web_chunks()
    if web_chunks:
        web_vector_store = Chroma.from_documents(
            web_chunks,
            embedding=embeddings,
            persist_directory=os.path.join(DB_DIR, "web_store")
        )
