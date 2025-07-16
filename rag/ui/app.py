import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'qa'))
from qa_engine import QAEngine, QAEngineConfig

st.set_page_config(page_title="Local RAG AI Assistant", page_icon="🤖", layout="centered")
st.title("📚 Local RAG AI Assistant")
st.markdown("Ask questions about your ingested documents. All processing is local and private.")

# Initialize QA Engine (singleton)
@st.cache_resource(show_spinner=False)
def get_qa_engine():
    config = QAEngineConfig(
        vector_store_path="D:/rag_system/vector_db",
        collection_name="documents",
        ollama_url="http://localhost:11434",
        model_name="mistral:7b",
        enable_logging=False
    )
    return QAEngine(config)

qa_engine = get_qa_engine()

# System status
with st.expander("System Status", expanded=False):
    status = qa_engine.get_system_status()
    st.write(f"**Engine Initialized:** {status['engine_initialized']}")
    st.write(f"**Components:** {status['components']}")
    st.write(f"**Vector DB Path:** {status['configuration']['vector_store_path']}")
    st.write(f"**Collection:** {status['configuration']['collection_name']}")
    st.write(f"**Ollama URL:** {status['configuration']['ollama_url']}")
    st.write(f"**Model:** {status['configuration']['model_name']}")
    st.write(f"**Timestamp:** {status['timestamp']}")

# User input
st.markdown("---")
question = st.text_input("Enter your question:", "What is artificial intelligence?")
ask = st.button("Ask")

if ask and question.strip():
    with st.spinner("Thinking..."):
        try:
            result = qa_engine.ask_question(question)
            st.markdown(f"### 🤖 Answer\n{result.answer}")
            st.markdown(f"**Confidence:** {result.confidence_score:.2f}")
            #st.markdown(f"**Sources:** {', '.join(result.sources) if result.sources else 'N/A'}")
            st.markdown(f"**Processing Time:** {result.processing_time:.2f}s")
        except Exception as e:
            st.error(f"Error: {e}\n\nIs Ollama running and the model loaded? Is the vector DB populated?")
            st.stop() 