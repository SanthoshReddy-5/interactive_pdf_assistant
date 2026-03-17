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

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None

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
        ["English", "Telugu", "Spanish", "French", "German", "Hindi", "Italian", "Portuguese"]
    )

    # Model name
    model_name = "llama3" 

# Main Interface
st.markdown(
    """
    <h1 style='color: #bf00ff;'>📄 Interactive PDF Assistant</h1>
    <p style='color: white;'>Upload a PDF document and interact with it seamlessly</p>
    """,
    unsafe_allow_html=True
)

def process_upload(uploaded_file):
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    chunks = rag_engine.process_pdf(tmp_file_path)
                    st.session_state.qa_chain = rag_engine.get_qa_chain()
                    st.session_state.document_chunks = chunks
                    # Automatically generate suggestions immediately after processing
                    st.session_state.suggested_questions = rag_engine.generate_suggestions(chunks)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success("PDF Processed Successfully!")
                except Exception as e:
                    error_msg = str(e)
                    if "Expected Embeddings to be non-empty" in error_msg:
                        st.warning("The uploaded PDF appears to be a scanned images document. Please upload a PDF that contains selectable text instead of scanned images.")
                    else:
                        st.error(f"An unexpected error occurred: {error_msg}")
                    st.session_state.current_file = None
                
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

process_upload(uploaded_file)

# Handling User Input (via Suggestions or Chat Input)
user_input = None

# Display Suggested Questions automatically
if st.session_state.suggested_questions:
    with st.expander("Suggested Questions", expanded=True):
        st.info("Here are 5 suggested questions based on your document:")
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
        if message["role"] == "user":
            st.markdown(f"<div style='color: #ff8c00;'>\n\n{message['content']}\n\n</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: #ffd400;'>\n\n{message['content']}\n\n</div>", unsafe_allow_html=True)
            
        if "audio" in message and message["audio"]:
            st.audio(message["audio"])

# Generate Response if last message is user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            prompt = st.session_state.messages[-1]["content"]
            
            try:
                response_stream = st.session_state.qa_chain.stream(
                    {"question": prompt, "language": language}
                )
                
                response_text = ""
                message_placeholder = st.empty()
                message_placeholder.markdown("<div style='color: #ffd400;'>\n\nAssistant is thinking...\n\n</div>", unsafe_allow_html=True)
                
                for chunk in response_stream:
                    response_text += chunk
                    message_placeholder.markdown(f"<div style='color: #ffd400;'>\n\n{response_text}▌\n\n</div>", unsafe_allow_html=True)
                
                # Final render without cursor
                message_placeholder.markdown(f"<div style='color: #ffd400;'>\n\n{response_text}\n\n</div>", unsafe_allow_html=True)
                
            except Exception as e:
                response_text = f"Error generating response: {e}"
                st.markdown(f"<div style='color: red;'>\n\n{response_text}\n\n</div>", unsafe_allow_html=True)
            
            # Audio
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