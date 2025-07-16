# Local RAG AI Assistant

A modular, fully local Retrieval-Augmented Generation (RAG) system for private document Q&A using [Ollama](https://ollama.com/), [LangChain](https://python.langchain.com/), and [ChromaDB](https://www.trychroma.com/). Ingest and search your own documents (PDF, TXT, DOCX, PPTX, MD) with a modern, privacy-first chat interface.

---

## Output Demo Video

Watch Demo [Click Here](https://drive.google.com/file/d/1XVxNRioGGvOZVXlPEPoU6YchgXfieihX/view?usp=sharing)

---

## Features

- **Local-first**: All processing and LLM inference run on your machine.
- **Multi-format ingestion**: PDF, TXT, DOCX, PPTX, Markdown.
- **Modular architecture**: Clean separation of ingestion, retrieval, QA, and UI.
- **Modern UI**: Streamlit-based chat interface.
- **Configurable**: Easily adapt paths, models, and settings.

---

## Project Structure

```
rag/
  ingestion/      # Document loading, chunking, embedding, vector DB
  retrieval/      # Vector search, context building
  qa/             # Query processing, answer generation, formatting
  ui/             # Streamlit chat app
tests/            # Test scripts
requirements.txt  # Python dependencies
README.md         # Project documentation
```

---

## Quick Start

### 1. **Install Requirements**

```bash
pip install -r requirements.txt
```

- Requires Python 3.9+.
- Install [Ollama](https://ollama.com/) and pull a model (e.g., `ollama pull mistral:7b`).

---

#### **Alternative: One-Click Dependency Installation**

You can install all required packages (with error handling and grouped output) using the provided `setup.py` script:

```bash
python setup.py
```

- This script will:
  - Upgrade `pip`
  - Install all core, AI/ML, interface, utility, and optional packages
  - Print a summary of any failed installations for manual follow-up

**Note:**
If you encounter any issues, check the summary at the end and manually install any failed packages as suggested.

---

### 2. **Ingest Your Documents**

- Place your documents in a folder (default: `D:/rag_system/data`).
- Edit paths in `rag/ingestion/document_processing.py` if needed.

Run the ingestion pipeline:
```bash
python rag/ingestion/document_processing.py
```
This will:
- Load and chunk documents
- Generate embeddings
- Store them in a local ChromaDB vector database

### 3. **Start Ollama**

Make sure Ollama is running and the desired model is loaded:
```bash
ollama run mistral:7b
```

### 4. **Launch the Chat UI**

```bash
streamlit run rag/ui/app.py
```
- The UI will connect to your local vector DB and Ollama instance.

---

## Technical Flow

1. **Ingestion** (`rag/ingestion/`)
   - Loads documents → splits into chunks → generates embeddings → stores in ChromaDB.
2. **Retrieval** (`rag/retrieval/`)
   - For a user query, retrieves relevant chunks using vector similarity.
3. **QA** (`rag/qa/`)
   - Builds context, sends to local LLM (via Ollama), formats the answer.
4. **UI** (`rag/ui/`)
   - Streamlit app for interactive Q&A.

---

## Configuration & Warnings

- **Path Customization Required**:  
  Several files use hardcoded paths (e.g., `D:/rag_system/data`, `C:/local_RAG_bot/local/vector_db`).  
  **You must update these paths** in:
  - `rag/ingestion/document_processing.py`
  - `rag/ingestion/src/vector_store.py`
  - `rag/qa/qa_engine.py`
  - `rag/retrieval/retrieval_engine.py`
  - `rag/ui/app.py`
- **Ollama**: Must be running locally with the correct model loaded.
- **Vector DB**: Ensure the vector DB path is consistent across modules.
- **Windows paths**: Paths are Windows-style by default; update for your OS as needed.

---

## Testing

- See `tests/` for setup and pipeline tests.
- Run tests with:
  ```bash
  python tests/test_setup.py
  ```

---

## Notes

- All code runs locally; no data leaves your machine.
- For best results, use high-quality, well-structured source documents.
- For advanced configuration, edit the config dictionaries/classes in each module.

---

