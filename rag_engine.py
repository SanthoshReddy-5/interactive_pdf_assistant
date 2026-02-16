import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class RAGEngine:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = ChatOllama(model=self.model_name)
        self.vectorstore = None
        self.retriever = None

    def process_pdf(self, file_path):
        """Loads and processes the PDF, creates vector store."""
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)
        
        # Create a temporary directory for ChromaDB to avoid locking issues or use in-memory
        # For simplicity in this session-based app, we can use an ephemeral client or a temp dir.
        # However, Chroma requires a persistent directory to be useful across re-runs if desired.
        # Here we use a simple in-memory approach or just let Chroma handle it.
        # Actually, for Streamlit, we want to persist it for the session.
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="pdf_rag"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        return chunks

    def get_qa_chain(self):
        """Sets up the RAG chain."""
        if not self.retriever:
            return None

        # Updated prompt to strictly enforce language
        template = """You are a helpful assistant.
Answer the question based ONLY on the following context.

Context:
{context}

Question: {question}

CRITICAL INSTRUCTION:
Answer strictly in the following language: {language}.
If the language is not English, do NOT providing any English translation or explanation. 
The entire response must be in {language} only.
"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {
                "context": itemgetter("question") | self.retriever, 
                "question": itemgetter("question"),
                "language": itemgetter("language") 
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def generate_suggestions(self, chunks):
        """Generates 5 sample questions based on the first few chunks."""
        if not chunks:
            return []
        
        # Take the first 2000 characters of text to generate questions from
        context_text = " ".join([chunk.page_content for chunk in chunks[:3]])[:2000]
        
        prompt_text = f"""Based on the following text, suggest exactly 5 questions that a user might ask.
        Return ONLY the questions, separated by a newline. Do not number them.
        
        Text:
        {context_text}
        """
        
        response = self.llm.invoke(prompt_text)
        # Parse response logic
        suggestions = [line.strip() for line in response.content.split('\n') if line.strip()]
        # Filter out numbering if the model adds it (e.g., "1. Question")
        clean_suggestions = []
        for s in suggestions:
            clean_s = s.lstrip('1234567890. -')
            if clean_s:
                clean_suggestions.append(clean_s)
        
        return clean_suggestions[:5]
