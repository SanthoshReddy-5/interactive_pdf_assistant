# 📄 Interactive PDF Assistant

> **A powerful, multilingual RAG (Retrieval-Augmented Generation) application built with Streamlit, LangChain, and Ollama.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange)

## 🌟 Overview

The **Interactive PDF Assistant** allows users to upload any PDF document and interact with it using natural language. It leverages local LLMs via **Ollama** to provide secure, offline-capable document analysis.

### 🏗️ Architecture Flowchart

```mermaid
graph TD
    A[User Uploads PDF] -->|PyMuPDF| B(Text Chunking)
    B -->|Sentence Transformers| C[(ChromaDB Vector Store)]
    D[User Asks Question] -->|Embed Query| C
    C -->|Retrieve Context| E[LangChain RAG Pipeline]
    E -->|Prompt + Context| F{Ollama local LLM}
    F -->|Generate Answer| G[Streamlit UI]
    G -->|Text Display| H[Custom Colored Chat]
    G -->|gTTS| I[Audio Voice Output]
```

Key capabilities include:
- **Multilingual Support**: Ask questions and get responses in **English, Telugu, Hindi, Spanish, French,** and more.
- **Voice Output**: Listen to the assistant's responses using integrated Text-to-Speech (TTS).
- **Smart Suggestions**: Automatically generates 5 relevant questions to help you start exploring the document.
- **Strict Language Enforcement**: Ensures responses are strictly in the requested language.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - For a clean, responsive web interface.
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) - Chains and retrieval logic.
- **Local LLM**: [Ollama](https://ollama.com/) - Running `llama3` (or other models) locally.
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) - Storing document embeddings.
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`).
- **PDF Parsing**: [PyMuPDF](https://pymupdf.readthedocs.io/) - Robust text extraction, including complex scripts.
- **Text-to-Speech**: [gTTS](https://gtts.readthedocs.io/) - Google Text-to-Speech integration.

## 🚀 Features

### 1. 📄 **Advanced PDF Processing**
Upload any PDF, and the app uses `PyMuPDF` to extract text with high fidelity, preserving structure and supporting non-Latin characters (e.g., Telugu, Hindi).

### 2. 🧠 **RAG Engine**
Uses a Retrieval-Augmented Generation pipeline to fetch relevant context chunks from the PDF and generate accurate answers using the local Ollama model.

### 3. ⚙️ **Purely Local Embeddings**
- Uses `all-MiniLM-L6-v2` downloaded via HuggingFace for lightning-fast, offline document vectorization.

### 4. 🌐 **Multilingual & Voice-Enabled**
- Select your preferred language from the sidebar.
- The assistant replies **textually** and **vocally** in that language.
- Custom Chat UI (User: Orange-Red, Assistant: Green-Yellow).

### 5. 💡 **Auto-Generated Questions**
Unsure what to ask? The assistant analyzes the document header/summary and suggests **5 intelligent questions** to kickstart your interaction.

## 📦 Installation

### Prerequisites
1. **Python 3.8+** installed.
2. **Ollama**: Download and install from [ollama.com](https://ollama.com).
   - Pull the default model:
     ```bash
     ollama pull llama3
     ```
   *(Note: You can use other models like `mistral` by changing the code in `app.py`)*

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/interactive-pdf-assistant.git
   cd interactive-pdf-assistant
   ```

2. **Install Dependencies**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   You can use the provided batch script (Windows) or run manually.
   - **Windows**: Double-click `start_app.bat`
   - **Manual**:
     ```bash
     streamlit run app.py
     ```

## 🎮 Usage Guide

1. **Start Ollama**: Ensure `ollama serve` is running in a terminal.
2. **Launch App**: Run the Streamlit app.
3. **Upload**: Use the sidebar to upload a PDF file.
4. **Select Language**: Choose your target language (e.g., Telugu).
5. **Interact**:
   - Click one of the **Suggested Questions**.
   - Or type your own question in the chat bar.
6. **Listen**: Click the audio player to hear the response.

## 📂 Project Structure

```
interactive_pdf_assistant/
├── app.py                 # Main Streamlit application entry point
├── rag_engine.py          # Core RAG logic (Loading, Splitting, Retrieval)
├── utils.py               # Utility functions (Text-to-Speech)
├── requirements.txt       # Python dependencies
├── start_app.bat          # Quick launch script for Windows
└── README.md              # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
