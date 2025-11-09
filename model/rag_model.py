import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from operator import itemgetter

# Initialize RAG model
def initialize_RAG():
    # Load PDF document
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "pdf", "Disaster_Preparedness_First_Aid_Handbook.pdf")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,    # Default is 1000, increased for better context and for testing
        chunk_overlap=200   # Default is 200
    )
    chunks = text_splitter.split_documents(docs)

    # Embeddings and vector store
    embeds = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeds
    )

    # Retrieval
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
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

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)
    

    # RAG Components for LCEL Chain
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Get response from RAG model
def get_response_RAG(question: str) -> str:
    rag_chain = initialize_RAG()
    response = rag_chain.invoke(
        {"question": question}
    )
    return response

# Debug from terminal
if __name__ == "__main__":
    print("DisasterAlertBot RAG Terminal Interface")
    print("Type your question below (type 'exit' to quit):\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting DisasterAlertBot. Stay safe!")
            break
        try:
            answer = get_response_RAG(user_input)
            print(f"Bot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")