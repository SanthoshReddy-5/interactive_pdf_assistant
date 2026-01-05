import os
import streamlit as st
from dotenv import load_dotenv

# import the required modules
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

# Streamlit Config
st.set_page_config(
    page_title="Interactive PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Interactive PDF Assistant")
st.write("Query your PDF using **LangChain** and **Streamlit** for Seamless Document Engagement.")

# Sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.markdown("---")
    st.write("### Tech Stack")
    st.write("- Groq")
    st.write("- LangChain")
    st.write("- FAISS")
    st.write("- Streamlit")
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []


# Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Processing
def process_pdf(pdf_file):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Create RAG Chain
def create_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
            You are a helpful AI assistant.
            Answer the user's question using ONLY the given context.
            If the answer is not present in the context, say "I don't know."

            Context:
            {context}

            Question:
            {question}
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# Handle PDF Upload
if uploaded_file and st.session_state.rag_chain is None:
    with st.spinner("Processing PDF..."):
        vectorstore = process_pdf(uploaded_file)
        st.session_state.rag_chain = create_rag_chain(vectorstore)
        st.success("PDF processed successfully! You can now ask questions.")

# Chat Input
user_question = st.chat_input("Ask a question about the PDF...")

if user_question and st.session_state.rag_chain:
    answer = st.session_state.rag_chain.invoke(user_question)

    st.session_state.chat_history.append(
        (user_question, answer)
    )

# Display Chat History
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)
