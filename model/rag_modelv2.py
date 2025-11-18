import os
from collections import deque
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import time

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILE = "Disaster_Preparedness_First_Aid_Handbook_Plaintext.pdf"
PDF_PATH = os.path.join(BASE_DIR, "pdf", PDF_FILE)
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

MAX_HISTORY_LENGTH = 5
chat_history = deque(maxlen=MAX_HISTORY_LENGTH)

TOP_K_CHUNKS = 2  # Only fetch top 2 relevant chunks for context

# -------------------- LOAD PDF --------------------
print("[DEBUG] Loading PDF...")
start_time = time.time()
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()
print(f"[DEBUG] Loaded {len(docs)} pages in {time.time() - start_time:.2f}s")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "?", "!"]
)
chunks = text_splitter.split_documents(docs)
print(f"[DEBUG] Split into {len(chunks)} chunks")

# -------------------- VECTOR STORE --------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("[DEBUG] Initializing vector store...")

start_time = time.time()
if not os.listdir(DB_DIR):
    vector_store = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    print("[DEBUG] Vector store created and persisted")
else:
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    print("[DEBUG] Vector store loaded from disk")
print(f"[DEBUG] Vector store ready in {time.time() - start_time:.2f}s")

# -------------------- LLM --------------------
llm = OllamaLLM(model="llama3.2:3b", temperature=0.2)
print("[DEBUG] LLM initialized")

# -------------------- PROMPT TEMPLATE --------------------
PROMPT_TEMPLATE = """
You are a disaster preparedness assistant.
Answer the question using the context provided. If the context contains partial information, answer as best as possible without inventing facts. 
If the context does not contain relevant information, respond: "Sorry, I don’t have enough information from my sources to answer that."

Chat History:
{chat_history}

Context:
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
    context = "\n\n".join([d.page_content.strip() for d in docs]) or "No relevant context found."
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

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    print("DisasterAlertBot (RAG + optional chat memory)")
    print("Type your question below (type 'exit' or 'quit' to quit):\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Stay safe!")
            break
        try:
            start = time.time()
            response = ask_question(user_input)
            print("Bot:", response)
            print(f"[DEBUG] Total time for this question: {time.time() - start:.2f}s\n")
        except Exception as e:
            print("Error:", e)
