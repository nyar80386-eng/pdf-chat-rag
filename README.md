import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="wide")

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fb;
    }
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    .answer-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 📄 DocChat AI")
st.markdown('<p class="subtitle">Apni PDF upload karo aur usse sawal poocho — AI turant jawab dega</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown("---")
    st.markdown("**Kaise Kaam Karta Hai:**")
    st.markdown("1. PDF upload karo\n2. AI document padhta hai\n3. Sawal poocho\n4. Turant jawab paao")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    uploaded_file = st.file_uploader("📎 Apni PDF upload karo", type="pdf")

    if uploaded_file is not None:
        with st.spinner("🔍 PDF process ho rahi hai..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM
