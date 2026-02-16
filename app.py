import streamlit as st
import os
import tempfile
from rag_engine import RAGEngine
from utils import text_to_speech

# Page Configuration
st.set_page_config(
    page_title="Interactive PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to get the engine
@st.cache_resource
def get_engine():
    return RAGEngine()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# We use the cached engine
rag_engine = get_engine()

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    # Language Selector
    language = st.selectbox(
        "Response Language",
        ["English", "Spanish", "French", "German", "Hindi", "Italian", "Portuguese", "Telugu"]
    )
    
    # Model Selector - Hidden/Fixed
    # Fixed to llama3 as requested to remove input
    # If we need to change model, we might need to invalidate cache or pass params to get_engine
    model_name = "llama3" 
    # For now, we assume fixed model or update the cached object (which is tricky with cache_resource)
    # If we strictly want to support changing models, we'd need get_engine(model_name)


# Main Interface
st.title("📄 Interactive PDF Assistant")

def process_upload(uploaded_file):
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                chunks = rag_engine.process_pdf(tmp_file_path)
                st.session_state.qa_chain = rag_engine.get_qa_chain()
                st.session_state.suggested_questions = rag_engine.generate_suggestions(chunks)
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
            st.success("PDF Processed Successfully!")

process_upload(uploaded_file)

# Handling User Input (via Suggestions or Chat Input)
user_input = None

# Display Suggested Questions in Expander
if st.session_state.suggested_questions:
    with st.expander("Show Suggested Questions", expanded=False):
        st.info("Suggested Questions based on your document:")
        # Display as a list or buttons. 
        # Using columns might be tight for 5 questions, let's use a vertical list of buttons
        for i, question in enumerate(st.session_state.suggested_questions):
            if st.button(question, key=f"suggestion_{i}"):
                user_input = question

# Chat Input
chat_input = st.chat_input("Ask a question about your PDF")
if chat_input:
    user_input = chat_input

# Process User Input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message and message["audio"]:
            st.audio(message["audio"])

# Generate Response if last message is user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt = st.session_state.messages[-1]["content"]
                
                # Contextualize language
                # We now pass language as a separate input variable to the chain
                
                try:
                    # Pass dictionary with question and language
                    response_text = st.session_state.qa_chain.invoke(
                        {"question": prompt, "language": language}
                    )
                except Exception as e:
                    response_text = f"Error generating response: {e}"

                st.markdown(response_text)
                
                # Audio
                # Pass language code for Telugu if selected
                audio_path = text_to_speech(response_text, language)
                if audio_path:
                    st.audio(audio_path)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "audio": audio_path
                })
    else:
        st.warning("Please upload a PDF first.")
        # Remove the user message since we can't answer it
        st.session_state.messages.pop()
